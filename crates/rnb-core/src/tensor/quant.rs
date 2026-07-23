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
