use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum QuantScheme {
    PerTensor,
    PerChannel(usize),
    PerGroup(usize),
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantScheme::PerTensor => write!(f, "per_tensor"),
            QuantScheme::PerChannel(axis) => write!(f, "per_channel({axis})"),
            QuantScheme::PerGroup(size) => write!(f, "per_group({size})"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantMeta {
    pub scheme: QuantScheme,
    pub scale: Vec<f32>,
    pub zero_point: Option<Vec<f32>>,
}

/// `.rnb` packed tensor storage type.
///
/// Dense packed variants identify runtime-ready packed weight layouts.
/// Raw variants identify GGUF-compatible tensor bytes stored in an `.rnb`
/// container without runtime repacking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantType {
    Q4K = 0,
    Q5K = 1,
    Q6K = 2,
    RawQ4K = 3,
    RawQ5_1 = 4,
    RawF32 = 5,
    RawQ2K = 6,
    RawQ5K = 7,
    RawQ6K = 8,
    RawQ8_0 = 9,
    RawQ2KTileGU = 10,
    RawBF16 = 11,
    Q80Pair = 12,
    Q4KCompact = 14,
}

impl QuantType {
    pub fn from_raw_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Q4K),
            1 => Some(Self::Q5K),
            2 => Some(Self::Q6K),
            3 => Some(Self::RawQ4K),
            4 => Some(Self::RawQ5_1),
            5 => Some(Self::RawF32),
            6 => Some(Self::RawQ2K),
            7 => Some(Self::RawQ5K),
            8 => Some(Self::RawQ6K),
            9 => Some(Self::RawQ8_0),
            10 => Some(Self::RawQ2KTileGU),
            11 => Some(Self::RawBF16),
            12 => Some(Self::Q80Pair),
            14 => Some(Self::Q4KCompact),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_scheme_display() {
        assert_eq!(format!("{}", QuantScheme::PerTensor), "per_tensor");
        assert_eq!(format!("{}", QuantScheme::PerChannel(0)), "per_channel(0)");
        assert_eq!(format!("{}", QuantScheme::PerGroup(128)), "per_group(128)");
    }
}
