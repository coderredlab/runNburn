use std::collections::HashSet;

use crate::error::LoaderError;
use crate::gguf::types::{GGMLType, GGUFValue, TensorInfo};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];
const MAX_STRING_LENGTH: usize = 64 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: usize = 4 * 1024 * 1024;
const MAX_ARRAY_NESTING: usize = 64;
const MAX_METADATA_KEY_LENGTH: usize = u16::MAX as usize;
const MAX_TENSOR_NAME_LENGTH: usize = 64;
const MIN_METADATA_RECORD_BYTES: usize = 13;
const MIN_TENSOR_INFO_BYTES: usize = 24;

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

    fn read_usize(&mut self, label: &str) -> Result<usize, LoaderError> {
        let offset = self.pos;
        let value = self.read_u64()?;
        usize::try_from(value).map_err(|_| LoaderError::ParseError {
            offset,
            msg: format!("{label} {value} does not fit usize"),
        })
    }

    /// GGUF 문자열: [len: u64][UTF-8 bytes]
    fn read_string(&mut self) -> Result<String, LoaderError> {
        let len = self.read_usize("string length")?;
        if len > MAX_STRING_LENGTH {
            return Err(LoaderError::ParseError {
                offset: self.pos - 8,
                msg: format!(
                    "string length {len} exceeds maximum supported length {MAX_STRING_LENGTH}"
                ),
            });
        }
        let offset = self.pos;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| LoaderError::ParseError {
                offset,
                msg: "invalid UTF-8 in string".to_string(),
            })
    }

    fn read_bool(&mut self) -> Result<bool, LoaderError> {
        let offset = self.pos;
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(LoaderError::ParseError {
                offset,
                msg: format!("invalid GGUF boolean value {value}; expected 0 or 1"),
            }),
        }
    }
}

fn read_value(r: &mut Reader<'_>, value_type: u32) -> Result<GGUFValue, LoaderError> {
    read_value_at_depth(r, value_type, 0)
}

fn read_value_at_depth(
    r: &mut Reader<'_>,
    value_type: u32,
    depth: usize,
) -> Result<GGUFValue, LoaderError> {
    if value_type > 12 {
        return Err(LoaderError::ParseError {
            offset: r.pos,
            msg: format!("unknown GGUF value type: {value_type}"),
        });
    }
    if depth > MAX_ARRAY_NESTING {
        return Err(LoaderError::ParseError {
            offset: r.pos,
            msg: format!("GGUF array nesting exceeds {MAX_ARRAY_NESTING} levels"),
        });
    }

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
            let elem_type = r.read_u32()?;
            if elem_type > 12 {
                return Err(LoaderError::ParseError {
                    offset: r.pos - 4,
                    msg: format!("unknown GGUF value type: {elem_type}"),
                });
            }
            let count = r.read_usize("array element count")?;
            if count > MAX_ARRAY_ELEMENTS {
                return Err(LoaderError::ParseError {
                    offset: r.pos - 8,
                    msg: format!(
                        "array element count {count} exceeds maximum {MAX_ARRAY_ELEMENTS}"
                    ),
                });
            }
            if count > r.remaining() {
                return Err(LoaderError::ParseError {
                    offset: r.pos,
                    msg: format!(
                        "array element count {count} exceeds {} remaining bytes",
                        r.remaining()
                    ),
                });
            }
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_value_at_depth(r, elem_type, depth + 1)?);
            }
            Ok(GGUFValue::Array(items))
        }
        10 => Ok(GGUFValue::U64(r.read_u64()?)),
        11 => Ok(GGUFValue::I64(r.read_i64()?)),
        12 => Ok(GGUFValue::F64(r.read_f64()?)),
        _ => unreachable!("GGUF value type range checked above"),
    }
}

fn metadata_alignment(metadata: &[(String, GGUFValue)]) -> Result<usize, LoaderError> {
    let Some((_, value)) = metadata.iter().find(|(key, _)| key == "general.alignment") else {
        return Ok(32);
    };

    let alignment = match value {
        GGUFValue::U8(value) => Some(usize::from(*value)),
        GGUFValue::I8(value) => usize::try_from(*value).ok(),
        GGUFValue::U16(value) => Some(usize::from(*value)),
        GGUFValue::I16(value) => usize::try_from(*value).ok(),
        GGUFValue::U32(value) => usize::try_from(*value).ok(),
        GGUFValue::I32(value) => usize::try_from(*value).ok(),
        GGUFValue::U64(value) => usize::try_from(*value).ok(),
        GGUFValue::I64(value) => usize::try_from(*value).ok(),
        _ => None,
    }
    .ok_or_else(|| LoaderError::ParseError {
        offset: 0,
        msg: "general.alignment must be a non-negative integer fitting usize".to_string(),
    })?;

    if !alignment.is_power_of_two() {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!("general.alignment {alignment} must be a positive power of two"),
        });
    }
    Ok(alignment)
}

impl GGUFFile {
    pub fn parse(data: &[u8]) -> Result<Self, LoaderError> {
        let mut r = Reader::new(data);

        // Magic
        let magic = r.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(LoaderError::InvalidMagic);
        }

        // GGUF v3 permits big-endian storage, but the runtime consumes mmap-backed
        // tensor payloads in little-endian GGML layout. Detect and reject it here
        // instead of reporting a bogus version or silently corrupting weights.
        let version_offset = r.pos;
        let version_bytes: [u8; 4] = r.read_bytes(4)?.try_into().unwrap();
        let version = u32::from_le_bytes(version_bytes);
        if !SUPPORTED_VERSIONS.contains(&version) {
            let big_endian_version = u32::from_be_bytes(version_bytes);
            if SUPPORTED_VERSIONS.contains(&big_endian_version) {
                return Err(LoaderError::ParseError {
                    offset: version_offset,
                    msg: "big-endian GGUF is not supported by the mmap tensor runtime".to_string(),
                });
            }
            return Err(LoaderError::UnsupportedVersion(version));
        }

        // Counts
        let tensor_count = r.read_usize("tensor count")?;
        let kv_count = r.read_usize("metadata count")?;
        let minimum_header_bytes = kv_count
            .checked_mul(MIN_METADATA_RECORD_BYTES)
            .and_then(|bytes| {
                tensor_count
                    .checked_mul(MIN_TENSOR_INFO_BYTES)
                    .and_then(|tensor_bytes| bytes.checked_add(tensor_bytes))
            })
            .ok_or_else(|| LoaderError::ParseError {
                offset: r.pos,
                msg: format!(
                    "tensor count {tensor_count} and metadata count {kv_count} overflow header size"
                ),
            })?;
        if minimum_header_bytes > r.remaining() {
            return Err(LoaderError::ParseError {
                offset: r.pos,
                msg: format!(
                    "tensor count {tensor_count} and metadata count {kv_count} cannot fit in {} remaining bytes",
                    r.remaining()
                ),
            });
        }

        // Metadata KV pairs
        let mut metadata = Vec::with_capacity(kv_count);
        let mut metadata_keys = HashSet::new();
        for _ in 0..kv_count {
            let key = r.read_string()?;
            if key.is_empty() || key.len() > MAX_METADATA_KEY_LENGTH || !key.is_ascii() {
                return Err(LoaderError::ParseError {
                    offset: r.pos - key.len(),
                    msg: format!(
                        "metadata key must be non-empty ASCII and at most {MAX_METADATA_KEY_LENGTH} bytes"
                    ),
                });
            }
            if !metadata_keys.insert(key.clone()) {
                return Err(LoaderError::ParseError {
                    offset: r.pos - key.len(),
                    msg: format!("duplicate metadata key '{key}'"),
                });
            }
            let value_type = r.read_u32()?;
            let value = read_value(&mut r, value_type)?;
            metadata.push((key, value));
        }

        let alignment = metadata_alignment(&metadata)?;

        // Tensor infos
        let mut tensor_infos = Vec::with_capacity(tensor_count);
        let mut tensor_names = HashSet::new();
        for _ in 0..tensor_count {
            let name = r.read_string()?;
            if name.len() >= MAX_TENSOR_NAME_LENGTH {
                return Err(LoaderError::ParseError {
                    offset: r.pos - name.len(),
                    msg: format!(
                        "tensor name is {} bytes; GGML requires fewer than {MAX_TENSOR_NAME_LENGTH}",
                        name.len()
                    ),
                });
            }
            if !tensor_names.insert(name.clone()) {
                return Err(LoaderError::ParseError {
                    offset: r.pos - name.len(),
                    msg: format!("duplicate tensor name '{name}'"),
                });
            }
            let n_dims = r.read_u32()? as usize;
            if n_dims > 4 {
                return Err(LoaderError::ParseError {
                    offset: r.pos - 4,
                    msg: format!(
                        "tensor '{name}' has {n_dims} dimensions; GGUF supports at most 4"
                    ),
                });
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.read_usize("tensor dimension")?);
            }
            // GGUF stores dims in reverse order (innermost first); normalize to row-major
            shape.reverse();
            let type_raw = r.read_u32()?;
            let ggml_type = GGMLType::try_from(type_raw)
                .map_err(|_| LoaderError::UnsupportedGGMLType(type_raw))?;
            let offset = r.read_u64()?;
            if offset % alignment as u64 != 0 {
                return Err(LoaderError::ParseError {
                    offset: r.pos - 8,
                    msg: format!(
                        "tensor '{name}' offset {offset} is not aligned to {alignment} bytes"
                    ),
                });
            }
            tensor_infos.push(TensorInfo {
                name,
                shape,
                ggml_type,
                offset,
            });
        }

        let remainder = r.pos % alignment;
        let padding = (alignment - remainder) % alignment;
        let data_start = r
            .pos
            .checked_add(padding)
            .ok_or_else(|| LoaderError::ParseError {
                offset: r.pos,
                msg: "tensor data offset overflows usize".to_string(),
            })?;
        if tensor_count > 0 && data_start > data.len() {
            return Err(LoaderError::ParseError {
                offset: r.pos,
                msg: format!(
                    "tensor data starts at {data_start}, beyond file length {}",
                    data.len()
                ),
            });
        }

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
        pub fn write_i64(&mut self, v: i64) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        pub fn write_f64(&mut self, v: f64) {
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

    #[test]
    fn test_parse_standard_64_bit_metadata_types() {
        let mut b = GGUFBuilder::with_counts(3, 0, 3);
        b.write_string("test.uint64");
        b.write_u32(10);
        b.write_u64(u64::MAX - 7);
        b.write_string("test.int64");
        b.write_u32(11);
        b.write_i64(-1_234_567_890_123);
        b.write_string("test.float64");
        b.write_u32(12);
        b.write_f64(std::f64::consts::PI);

        let gguf = GGUFFile::parse(&b.build()).expect("standard 64-bit values should parse");
        assert_eq!(gguf.metadata[0].1, GGUFValue::U64(u64::MAX - 7));
        assert_eq!(gguf.metadata[1].1, GGUFValue::I64(-1_234_567_890_123));
        assert_eq!(gguf.metadata[2].1, GGUFValue::F64(std::f64::consts::PI));
    }

    #[test]
    fn test_parse_rejects_invalid_bool_encoding() {
        let mut b = GGUFBuilder::with_counts(3, 0, 1);
        b.write_string("test.bool");
        b.write_u32(7);
        b.write_u8(2);

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("boolean")
        ));
    }

    #[test]
    fn test_parse_rejects_non_power_of_two_alignment() {
        let mut b = GGUFBuilder::with_counts(3, 0, 1);
        b.write_string("general.alignment");
        b.write_u32(4);
        b.write_u32(24);

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("power of two")
        ));
    }

    #[test]
    fn test_parse_rejects_invalid_alignment() {
        let mut b = GGUFBuilder::with_counts(3, 0, 1);
        b.write_string("general.alignment");
        b.write_u32(4);
        b.write_u32(12);

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("alignment")
        ));
    }

    #[test]
    fn test_parse_rejects_big_endian_explicitly() {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&3u32.to_be_bytes());
        data.extend_from_slice(&0u64.to_be_bytes());
        data.extend_from_slice(&0u64.to_be_bytes());

        assert!(matches!(
            GGUFFile::parse(&data),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("big-endian")
        ));
    }

    #[test]
    fn test_parse_rejects_impossible_header_counts_before_allocation() {
        let data = GGUFBuilder::with_counts(3, 0, u64::MAX).build();
        assert!(matches!(
            GGUFFile::parse(&data),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("metadata count")
        ));
    }

    #[test]
    fn test_parse_rejects_oversized_string_before_reading_payload() {
        let mut b = GGUFBuilder::with_counts(3, 0, 1);
        b.write_string("test.string");
        b.write_u32(8);
        b.write_u64((MAX_STRING_LENGTH + 1) as u64);

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("string length")
        ));
    }

    #[test]
    fn test_parse_rejects_oversized_array_before_allocation() {
        let mut b = GGUFBuilder::with_counts(3, 0, 1);
        b.write_string("test.array");
        b.write_u32(9);
        b.write_u32(0);
        b.write_u64((MAX_ARRAY_ELEMENTS + 1) as u64);

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("array element count")
        ));
    }

    #[test]
    fn test_parse_rejects_empty_array_with_unknown_element_type() {
        let mut b = GGUFBuilder::with_counts(3, 0, 1);
        b.write_string("test.array");
        b.write_u32(9);
        b.write_u32(99);
        b.write_u64(0);

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("unknown GGUF value type")
        ));
    }

    #[test]
    fn test_parse_rejects_duplicate_metadata_keys() {
        let mut b = GGUFBuilder::with_counts(3, 0, 2);
        for value in [1, 2] {
            b.write_string("test.key");
            b.write_u32(0);
            b.write_u8(value);
        }

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("duplicate metadata key")
        ));
    }

    #[test]
    fn test_parse_rejects_duplicate_tensor_names() {
        let mut b = GGUFBuilder::with_counts(3, 2, 0);
        for offset in [0, 32] {
            b.write_string("duplicate.weight");
            b.write_u32(1);
            b.write_u64(1);
            b.write_u32(0);
            b.write_u64(offset);
        }

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("duplicate tensor name")
        ));
    }

    #[test]
    fn test_parse_rejects_tensor_name_at_ggml_limit() {
        let mut b = GGUFBuilder::with_counts(3, 1, 0);
        b.write_string(&"x".repeat(64));
        b.write_u32(1);
        b.write_u64(1);
        b.write_u32(0);
        b.write_u64(0);

        assert!(matches!(
            GGUFFile::parse(&b.build()),
            Err(LoaderError::ParseError { msg, .. }) if msg.contains("tensor name")
        ));
    }
}
