use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
    U8,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 | DType::I4 => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I8 => write!(f, "i8"),
            DType::I4 => write!(f, "i4"),
            DType::U8 => write!(f, "u8"),
        }
    }
}

pub trait TensorElement: Copy + Send + Sync + 'static {
    const DTYPE: DType;
}

impl TensorElement for f32 {
    const DTYPE: DType = DType::F32;
}

impl TensorElement for u8 {
    const DTYPE: DType = DType::U8;
}

impl TensorElement for i8 {
    const DTYPE: DType = DType::I8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::I8.size_bytes(), 1);
        assert_eq!(DType::U8.size_bytes(), 1);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::I4), "i4");
    }

    #[test]
    fn test_tensor_element_dtype() {
        assert_eq!(f32::DTYPE, DType::F32);
        assert_eq!(u8::DTYPE, DType::U8);
        assert_eq!(i8::DTYPE, DType::I8);
    }
}
