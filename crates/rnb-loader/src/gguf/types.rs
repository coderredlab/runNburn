/// GGUF 텐서 데이터 타입 (llama.cpp ggml_type 기준)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    BF16 = 30,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 16,
    IQ3_XXS = 18,
    IQ2_S = 22,
    IQ4_XS = 23,
    I32 = 26,
}

impl TryFrom<u32> for GGMLType {
    type Error = u32;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            30 => Ok(Self::BF16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            16 => Ok(Self::IQ2_XXS),
            18 => Ok(Self::IQ3_XXS),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            26 => Ok(Self::I32),
            other => Err(other),
        }
    }
}

/// GGUF 메타데이터 값 타입
#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}

/// GGUF 파일 내 텐서 정보 레코드
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: GGMLType,
    /// 텐서 데이터 시작 오프셋 (data 영역 기준, 헤더 기준 아님)
    pub offset: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_round_trip() {
        assert_eq!(GGMLType::try_from(0u32), Ok(GGMLType::F32));
        assert_eq!(GGMLType::try_from(1u32), Ok(GGMLType::F16));
        assert_eq!(GGMLType::try_from(30u32), Ok(GGMLType::BF16));
        assert_eq!(GGMLType::try_from(8u32), Ok(GGMLType::Q8_0));
        assert_eq!(GGMLType::try_from(14u32), Ok(GGMLType::Q6_K));
        assert_eq!(GGMLType::try_from(23u32), Ok(GGMLType::IQ4_XS));
        assert_eq!(GGMLType::try_from(26u32), Ok(GGMLType::I32));
        assert!(GGMLType::try_from(99u32).is_err());
    }

    #[test]
    fn test_gguf_value_variants() {
        let v = GGUFValue::F32(1.5);
        assert!(matches!(v, GGUFValue::F32(_)));
        let arr = GGUFValue::Array(vec![GGUFValue::U32(1), GGUFValue::U32(2)]);
        assert!(matches!(arr, GGUFValue::Array(_)));
    }

    #[test]
    fn test_tensor_info_fields() {
        let ti = TensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![4096, 4096],
            ggml_type: GGMLType::F16,
            offset: 0,
        };
        assert_eq!(ti.shape.len(), 2);
        assert_eq!(ti.ggml_type, GGMLType::F16);
    }
}
