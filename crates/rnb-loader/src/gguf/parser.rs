use crate::error::LoaderError;
use crate::gguf::types::{GGMLType, GGUFValue, TensorInfo};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

pub struct GGUFFile {
    pub version: u32,
    pub metadata: Vec<(String, GGUFValue)>,
    pub tensor_infos: Vec<TensorInfo>,
    /// 파일 내 텐서 데이터 영역의 시작 바이트 오프셋
    pub data_start: usize,
    pub alignment: usize,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_bytes(&mut self, n: usize) -> Result<&[u8], LoaderError> {
        if self.remaining() < n {
            return Err(LoaderError::ParseError {
                offset: self.pos,
                msg: format!("need {n} bytes, only {} remain", self.remaining()),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, LoaderError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, LoaderError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, LoaderError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> Result<i16, LoaderError> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, LoaderError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, LoaderError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, LoaderError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, LoaderError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, LoaderError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, LoaderError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }

    /// GGUF 문자열: [len: u64][UTF-8 bytes]
    fn read_string(&mut self) -> Result<String, LoaderError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?.to_vec();
        String::from_utf8(bytes).map_err(|_| LoaderError::ParseError {
            offset: self.pos,
            msg: "invalid UTF-8 in string".to_string(),
        })
    }

    fn read_bool(&mut self) -> Result<bool, LoaderError> {
        Ok(self.read_u8()? != 0)
    }
}

fn read_value(r: &mut Reader<'_>, value_type: u32) -> Result<GGUFValue, LoaderError> {
    match value_type {
        0 => Ok(GGUFValue::U8(r.read_u8()?)),
        1 => Ok(GGUFValue::I8(r.read_i8()?)),
        2 => Ok(GGUFValue::U16(r.read_u16()?)),
        3 => Ok(GGUFValue::I16(r.read_i16()?)),
        4 => Ok(GGUFValue::U32(r.read_u32()?)),
        5 => Ok(GGUFValue::I32(r.read_i32()?)),
        6 => Ok(GGUFValue::F32(r.read_f32()?)),
        7 => Ok(GGUFValue::Bool(r.read_bool()?)),
        8 => Ok(GGUFValue::String(r.read_string()?)),
        9 => {
            // Array: [elem_type: u32][count: u64][elements...]
            let elem_type = r.read_u32()?;
            let count = r.read_u64()? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_value(r, elem_type)?);
            }
            Ok(GGUFValue::Array(items))
        }
        10 => Ok(GGUFValue::I64(r.read_i64()?)),
        11 => Ok(GGUFValue::F64(r.read_f64()?)),
        12 => Ok(GGUFValue::U64(r.read_u64()?)),
        other => Err(LoaderError::ParseError {
            offset: 0,
            msg: format!("unknown GGUF value type: {other}"),
        }),
    }
}

impl GGUFFile {
    pub fn parse(data: &[u8]) -> Result<Self, LoaderError> {
        let mut r = Reader::new(data);

        // Magic
        let magic = r.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(LoaderError::InvalidMagic);
        }

        // Version
        let version = r.read_u32()?;
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(LoaderError::UnsupportedVersion(version));
        }

        // Counts
        let tensor_count = r.read_u64()? as usize;
        let kv_count = r.read_u64()? as usize;

        // Metadata KV pairs
        let mut metadata = Vec::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.read_string()?;
            let value_type = r.read_u32()?;
            let value = read_value(&mut r, value_type)?;
            metadata.push((key, value));
        }

        // alignment from metadata (default: 32)
        // general.alignment는 U32로 저장되는 게 보통이지만
        // 일부 파일은 U64나 I32로 저장할 수 있어 모든 정수 타입을 처리한다.
        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| match v {
                GGUFValue::U8(a) => Some(*a as usize),
                GGUFValue::I8(a) => Some(*a as usize),
                GGUFValue::U16(a) => Some(*a as usize),
                GGUFValue::I16(a) => Some(*a as usize),
                GGUFValue::U32(a) => Some(*a as usize),
                GGUFValue::I32(a) => Some(*a as usize),
                GGUFValue::U64(a) => Some(*a as usize),
                GGUFValue::I64(a) => Some(*a as usize),
                _ => None,
            })
            .filter(|&a| a.is_power_of_two() && a >= 8)
            .unwrap_or(32);

        // Tensor infos
        let mut tensor_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.read_string()?;
            let n_dims = r.read_u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.read_u64()? as usize);
            }
            // GGUF stores dims in reverse order (innermost first); normalize to row-major
            shape.reverse();
            let type_raw = r.read_u32()?;
            let ggml_type = GGMLType::try_from(type_raw)
                .map_err(|_| LoaderError::UnsupportedGGMLType(type_raw))?;
            let offset = r.read_u64()?;
            tensor_infos.push(TensorInfo {
                name,
                shape,
                ggml_type,
                offset,
            });
        }

        // Align to `alignment`
        let remainder = r.pos % alignment;
        let data_start = if remainder == 0 {
            r.pos
        } else {
            r.pos + (alignment - remainder)
        };

        Ok(GGUFFile {
            version,
            metadata,
            tensor_infos,
            data_start,
            alignment,
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// 테스트용 최소 GGUF 바이너리 빌더
    pub struct GGUFBuilder {
        pub buf: Vec<u8>,
    }

    impl GGUFBuilder {
        pub fn new(version: u32) -> Self {
            let mut b = Self { buf: Vec::new() };
            b.write_bytes(b"GGUF");
            b.write_u32(version);
            b
        }

        pub fn write_bytes(&mut self, bytes: &[u8]) {
            self.buf.extend_from_slice(bytes);
        }
        pub fn write_u8(&mut self, v: u8) {
            self.buf.push(v);
        }
        pub fn write_u32(&mut self, v: u32) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        pub fn write_u64(&mut self, v: u64) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        pub fn write_string(&mut self, s: &str) {
            self.write_u64(s.len() as u64);
            self.write_bytes(s.as_bytes());
        }

        pub fn with_counts(version: u32, tensor_count: u64, kv_count: u64) -> Self {
            let mut b = Self::new(version);
            b.write_u64(tensor_count);
            b.write_u64(kv_count);
            b
        }

        pub fn build(self) -> Vec<u8> {
            self.buf
        }
    }

    /// 메타데이터 1개 (string KV), 텐서 0개인 최소 GGUF
    pub fn make_minimal_gguf(arch: &str) -> Vec<u8> {
        let mut b = GGUFBuilder::with_counts(3, 0, 1);
        // KV: "general.architecture" = String(arch)
        b.write_string("general.architecture");
        b.write_u32(8); // value type = String
        b.write_string(arch);
        // pad to 32 bytes alignment from current position
        let current = b.buf.len();
        let pad = (32 - (current % 32)) % 32;
        for _ in 0..pad {
            b.write_u8(0);
        }
        b.build()
    }

    pub fn make_gguf_with_tensor(arch: &str) -> Vec<u8> {
        let mut b = GGUFBuilder::with_counts(3, 1, 1);
        // KV
        b.write_string("general.architecture");
        b.write_u32(8);
        b.write_string(arch);
        // TensorInfo: name="token_embd.weight", shape=[8,4] (stored reversed: [4,8]), F32, offset=0
        b.write_string("token_embd.weight");
        b.write_u32(2); // n_dims
        b.write_u64(4); // dim0 (innermost → becomes shape[1] after reverse)
        b.write_u64(8); // dim1 (outermost → becomes shape[0] after reverse)
        b.write_u32(0); // GGMLType::F32
        b.write_u64(0); // offset
                        // align to 32
        let current = b.buf.len();
        let pad = (32 - (current % 32)) % 32;
        for _ in 0..pad {
            b.write_u8(0);
        }
        // tensor data: 8*4*4 = 128 bytes of zeros
        b.write_bytes(&vec![0u8; 128]);
        b.build()
    }

    #[test]
    fn test_parse_invalid_magic() {
        let data = b"XXXX\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert!(matches!(
            GGUFFile::parse(data),
            Err(LoaderError::InvalidMagic)
        ));
    }

    #[test]
    fn test_parse_unsupported_version() {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&1u32.to_le_bytes()); // version 1 = unsupported
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        data.extend_from_slice(&0u64.to_le_bytes()); // kv count
        assert!(matches!(
            GGUFFile::parse(&data),
            Err(LoaderError::UnsupportedVersion(1))
        ));
    }

    #[test]
    fn test_parse_minimal_gguf_v3() {
        let data = make_minimal_gguf("llama");
        let gguf = GGUFFile::parse(&data).expect("should parse");
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensor_infos.len(), 0);
        assert_eq!(gguf.metadata.len(), 1);
        assert_eq!(gguf.metadata[0].0, "general.architecture");
        assert!(matches!(&gguf.metadata[0].1, GGUFValue::String(s) if s == "llama"));
    }

    #[test]
    fn test_parse_tensor_info() {
        let data = make_gguf_with_tensor("llama");
        let gguf = GGUFFile::parse(&data).expect("should parse");
        assert_eq!(gguf.tensor_infos.len(), 1);
        let ti = &gguf.tensor_infos[0];
        assert_eq!(ti.name, "token_embd.weight");
        // shape reversed from [4, 8] → [8, 4]
        assert_eq!(ti.shape, vec![8, 4]);
        assert_eq!(ti.ggml_type, GGMLType::F32);
        assert_eq!(ti.offset, 0);
    }

    #[test]
    fn test_data_start_aligned() {
        let data = make_minimal_gguf("llama");
        let gguf = GGUFFile::parse(&data).expect("should parse");
        assert_eq!(gguf.data_start % 32, 0);
    }
}
