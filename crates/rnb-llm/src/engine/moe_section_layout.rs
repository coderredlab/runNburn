#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MoeSectionGateUpLayout {
    Q4KPair,
    UnpackedScales,
    ScalePlane,
}

impl MoeSectionGateUpLayout {
    #[inline]
    pub(super) fn unit_size(self) -> usize {
        super::moe_section_dispatch::gate_up_unit_size(self)
    }

    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    #[inline]
    pub(super) fn uses_scale_plane(self) -> bool {
        matches!(self, MoeSectionGateUpLayout::ScalePlane)
    }
}

impl From<rnb_core::rnb_moe::MoeSectionGateUpLayout> for MoeSectionGateUpLayout {
    #[inline]
    fn from(layout: rnb_core::rnb_moe::MoeSectionGateUpLayout) -> Self {
        match layout {
            rnb_core::rnb_moe::MoeSectionGateUpLayout::Q4KPair => Self::Q4KPair,
            rnb_core::rnb_moe::MoeSectionGateUpLayout::UnpackedScales => Self::UnpackedScales,
            rnb_core::rnb_moe::MoeSectionGateUpLayout::ScalePlane => Self::ScalePlane,
        }
    }
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) fn moe_section_gate_up_layout(gate_up_quant: u8) -> Option<MoeSectionGateUpLayout> {
    rnb_core::rnb_moe::MoeSectionGateUpLayout::from_section_quant(gate_up_quant).map(Into::into)
}

#[allow(dead_code)]
pub(super) fn moe_section_gate_up_unit_size(gate_up_quant: u8) -> Option<usize> {
    moe_section_gate_up_layout(gate_up_quant).map(MoeSectionGateUpLayout::unit_size)
}
