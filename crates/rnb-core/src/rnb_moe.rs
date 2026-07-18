//! MoE/section `.rnb` container format.
//!
//! Layout:
//!   [magic(4) = "RNBM"]
//!   [version(4) = 2, little-endian]
//!   [section_count(4), little-endian]
//!   [section_count * { id(1), offset(8), size(8) }]  // 17 bytes/entry
//!   ... (section bodies follow at their declared offsets)

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub sections: Vec<SectionTableEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionTableEntry {
    pub id: SectionId,
    pub offset: u64,
    pub size: u64,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionId {
    PrefillPacked = 0x01,
    MoeDecode = 0x02,
    AttnDecode = 0x03,
    GdnDecode = 0x04,
    KvSchema = 0x05,
}

impl SectionId {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::PrefillPacked),
            0x02 => Some(Self::MoeDecode),
            0x03 => Some(Self::AttnDecode),
            0x04 => Some(Self::GdnDecode),
            0x05 => Some(Self::KvSchema),
            _ => None,
        }
    }
}

pub const MOE_MAGIC: [u8; 4] = *b"RNBM";
pub const MOE_VERSION: u32 = 2;
pub const MOE_HEADER_FIXED_LEN: usize = 12; // magic(4) + version(4) + count(4)
pub const MOE_SECTION_ENTRY_LEN: usize = 17; // id(1) + offset(8) + size(8)

pub const GATE_UP_QUANT_Q4K_PAIR: u8 = 0x12;
pub const GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES: u8 = 0x32;
pub const GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE: u8 = 0x33;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeSectionGateUpLayout {
    Q4KPair,
    UnpackedScales,
    ScalePlane,
}

impl MoeSectionGateUpLayout {
    #[inline]
    pub const fn from_section_quant(gate_up_quant: u8) -> Option<Self> {
        match gate_up_quant {
            GATE_UP_QUANT_Q4K_PAIR => Some(Self::Q4KPair),
            GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES => Some(Self::UnpackedScales),
            GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE => Some(Self::ScalePlane),
            _ => None,
        }
    }
}

impl MoeHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(MOE_HEADER_FIXED_LEN + self.sections.len() * MOE_SECTION_ENTRY_LEN);
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        for s in &self.sections {
            out.push(s.id as u8);
            out.extend_from_slice(&s.offset.to_le_bytes());
            out.extend_from_slice(&s.size.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < MOE_HEADER_FIXED_LEN {
            return Err(format!("header too short: {}", bytes.len()));
        }
        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != MOE_MAGIC {
            return Err(format!("invalid magic: {:?}", magic));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != MOE_VERSION {
            return Err(format!("unsupported version: {}", version));
        }
        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let expected_len = MOE_HEADER_FIXED_LEN + count * MOE_SECTION_ENTRY_LEN;
        if bytes.len() < expected_len {
            return Err(format!("truncated: {} < {}", bytes.len(), expected_len));
        }
        let mut sections = Vec::with_capacity(count);
        for i in 0..count {
            let base = MOE_HEADER_FIXED_LEN + i * MOE_SECTION_ENTRY_LEN;
            let id_byte = bytes[base];
            let id = SectionId::from_u8(id_byte)
                .ok_or_else(|| format!("unknown section id: {:#x}", id_byte))?;
            let offset = u64::from_le_bytes(bytes[base + 1..base + 9].try_into().unwrap());
            let size = u64::from_le_bytes(bytes[base + 9..base + 17].try_into().unwrap());
            sections.push(SectionTableEntry { id, offset, size });
        }
        Ok(Self {
            magic,
            version,
            sections,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_header_roundtrip_multi_section() {
        let h = MoeHeader {
            magic: *b"RNBM",
            version: 2,
            sections: vec![
                SectionTableEntry {
                    id: SectionId::MoeDecode,
                    offset: 128,
                    size: 4096,
                },
                SectionTableEntry {
                    id: SectionId::KvSchema,
                    offset: 4224,
                    size: 16,
                },
                SectionTableEntry {
                    id: SectionId::AttnDecode,
                    offset: 4240,
                    size: 2048,
                },
            ],
        };
        let bytes = h.to_bytes();
        let back = MoeHeader::from_bytes(&bytes).expect("parse");
        assert_eq!(h, back);
    }

    #[test]
    fn moe_header_rejects_wrong_magic() {
        let bytes = b"WRONG\0\0\0\x02\0\0\0\0\0\0\0".to_vec();
        assert!(MoeHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn moe_header_rejects_wrong_version() {
        let mut bytes = b"RNBM".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        assert!(MoeHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn moe_header_rejects_unknown_section_id() {
        let mut bytes = b"RNBM".to_vec();
        bytes.extend_from_slice(&2u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1u32.to_le_bytes()); // count
        bytes.push(0xFE); // unknown id
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        assert!(MoeHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn gate_up_layout_classifies_section_quant_tags() {
        assert_eq!(
            MoeSectionGateUpLayout::from_section_quant(GATE_UP_QUANT_Q4K_PAIR),
            Some(MoeSectionGateUpLayout::Q4KPair)
        );
        assert_eq!(
            MoeSectionGateUpLayout::from_section_quant(GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES),
            Some(MoeSectionGateUpLayout::UnpackedScales)
        );
        assert_eq!(
            MoeSectionGateUpLayout::from_section_quant(GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE),
            Some(MoeSectionGateUpLayout::ScalePlane)
        );
        assert_eq!(MoeSectionGateUpLayout::from_section_quant(0xFF), None);
    }
}
