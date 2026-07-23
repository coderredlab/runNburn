use std::collections::HashMap;
use std::sync::Arc;

use crate::error::LoaderError;
use crate::gguf::parser::GGUFFile;
use crate::gguf::types::GGMLType;
use rnb_core::tensor::dtype::DType;
use rnb_core::tensor::storage::Storage;
use rnb_core::tensor::tensor::Tensor;

/// GGUF GGMLType → rnb-core DType
/// 양자화 타입은 raw byte 보존을 위해 U8로 매핑
pub fn ggml_type_to_dtype(t: GGMLType) -> DType {
    match t {
        GGMLType::F32 => DType::F32,
        GGMLType::F16 | GGMLType::BF16 => DType::F16,
        GGMLType::Q8_0 | GGMLType::Q8_1 => DType::I8,
        _ => DType::U8, // Q4_x, Q5_x, Q2_K .. Q6_K → raw bytes
    }
}

/// 텐서의 실제 바이트 크기 계산 (양자화 블록 구조 반영)
pub fn compute_tensor_size(shape: &[usize], ggml_type: GGMLType) -> usize {
    let numel: usize = shape.iter().product();
    // (block_size, type_size_bytes)
    let (block_size, type_bytes) = ggml_quant_params(ggml_type);
    // numel must be divisible by block_size for quantized types
    let blocks = numel.div_ceil(block_size);
    blocks * type_bytes
}

/// llama.cpp 기준 양자화 파라미터: (elements per block, bytes per block)
pub fn ggml_quant_params(t: GGMLType) -> (usize, usize) {
    match t {
        GGMLType::F32 => (1, 4),
        GGMLType::F16 | GGMLType::BF16 => (1, 2),
        GGMLType::I32 => (1, 4),
        GGMLType::I8 => (1, 1),
        GGMLType::I16 => (1, 2),
        GGMLType::I64 | GGMLType::F64 => (1, 8),
        GGMLType::Q4_0 => (32, 18),
        GGMLType::Q4_1 => (32, 20),
        GGMLType::Q5_0 => (32, 22),
        GGMLType::Q5_1 => (32, 24),
        GGMLType::Q8_0 => (32, 34),
        GGMLType::Q8_1 => (32, 36),
        GGMLType::Q2_K => (256, 84),
        GGMLType::Q3_K => (256, 110),
        GGMLType::Q4_K => (256, 144),
        GGMLType::Q5_K => (256, 176),
        GGMLType::Q6_K => (256, 210),
        GGMLType::Q8_K => (256, 292),
        GGMLType::IQ2_XXS => (256, 66),
        GGMLType::IQ2_XS => (256, 74),
        GGMLType::IQ3_XXS => (256, 98),
        GGMLType::IQ1_S => (256, 50),
        GGMLType::IQ4_NL => (32, 18),
        GGMLType::IQ3_S => (256, 110),
        GGMLType::IQ2_S => (256, 82),
        GGMLType::IQ4_XS => (256, 136),
        GGMLType::IQ1_M => (256, 56),
        GGMLType::TQ1_0 => (256, 54),
        GGMLType::TQ2_0 => (256, 66),
        GGMLType::MXFP4 => (32, 17),
        GGMLType::NVFP4 => (64, 36),
        GGMLType::Q1_0 => (128, 18),
        GGMLType::Q2_0 => (64, 18),
    }
}

/// GGUF 파일의 모든 텐서를 mmap 기반 zero-copy Tensor로 변환한다.
///
/// `mmap`의 소유권을 받아 `Arc<Storage::Mmap>` 으로 한 번만 감싸고,
/// 각 텐서는 `Arc::clone` + byte offset 뷰로 만든다.
///
/// # TODO
/// 현재 구현은 mmap 전체를 Storage::Mmap으로 보유하고 각 텐서는 byte offset 뷰를 참조한다.
/// 실제 zero-copy를 위해 Storage::Mmap의 as_slice()를 offset 포함해 접근해야 함.
/// GGUF 파일의 모든 텐서를 mmap 기반 zero-copy Tensor로 변환한다.
///
/// 반환값:
/// - tensors: 이름 → Tensor (양자화 타입은 [byte_count] 1D U8/I8 텐서)
/// - float_shapes: 이름 → 원래 float shape (양자화 타입에만 존재)
/// - ggml_types: 이름 → 원래 GGMLType (모든 텐서)
/// - file_offsets: 이름 → GGUF 파일 내 절대 byte offset
pub fn map_tensors(
    gguf: &GGUFFile,
    mmap: rnb_core::tensor::FileMmapStorage,
) -> Result<
    (
        HashMap<String, Tensor>,
        HashMap<String, Vec<usize>>,
        HashMap<String, GGMLType>,
        HashMap<String, usize>,
    ),
    LoaderError,
> {
    let storage = Arc::new(Storage::FileMmap(mmap));

    let mut tensors = HashMap::new();
    let mut float_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut ggml_types: HashMap<String, GGMLType> = HashMap::new();
    let mut file_offsets: HashMap<String, usize> = HashMap::new();

    for info in &gguf.tensor_infos {
        let dtype = ggml_type_to_dtype(info.ggml_type);
        let byte_offset = gguf.data_start + info.offset as usize;

        // 양자화 타입의 경우 실제 바이트 크기와 float numel이 다름.
        // 따라서 양자화 타입은 [actual_bytes] 1D로 저장하고,
        // 원래 float shape은 float_shapes에 따로 보존한다.
        let (mmap_shape, actual_dtype) = match dtype {
            DType::U8 | DType::I8 => {
                let actual_bytes = compute_tensor_size(&info.shape, info.ggml_type);
                float_shapes.insert(info.name.clone(), info.shape.clone());
                (vec![actual_bytes], dtype)
            }
            _ => (info.shape.clone(), dtype),
        };

        let tensor =
            Tensor::from_mmap(Arc::clone(&storage), byte_offset, &mmap_shape, actual_dtype)
                .map_err(LoaderError::CoreError)?;
        ggml_types.insert(info.name.clone(), info.ggml_type);
        file_offsets.insert(info.name.clone(), byte_offset);
        tensors.insert(info.name.clone(), tensor);
    }
    Ok((tensors, float_shapes, ggml_types, file_offsets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::types::GGMLType;
    use rnb_core::tensor::dtype::DType;

    #[test]
    fn test_ggml_type_to_dtype_f32() {
        assert_eq!(ggml_type_to_dtype(GGMLType::F32), DType::F32);
    }

    #[test]
    fn test_ggml_type_to_dtype_f16() {
        assert_eq!(ggml_type_to_dtype(GGMLType::F16), DType::F16);
    }

    #[test]
    fn test_ggml_type_to_dtype_q8_0() {
        assert_eq!(ggml_type_to_dtype(GGMLType::Q8_0), DType::I8);
    }

    #[test]
    fn test_ggml_type_to_dtype_q4_0() {
        assert_eq!(ggml_type_to_dtype(GGMLType::Q4_0), DType::U8);
    }

    #[test]
    fn test_compute_tensor_size_f32() {
        // shape [4, 4] = 16 elements * 4 bytes = 64
        assert_eq!(compute_tensor_size(&[4, 4], GGMLType::F32), 64);
        assert_eq!(compute_tensor_size(&[4, 4], GGMLType::I32), 64);
    }

    #[test]
    fn test_compute_tensor_size_f16() {
        // 16 elements * 2 bytes = 32
        assert_eq!(compute_tensor_size(&[4, 4], GGMLType::F16), 32);
    }

    #[test]
    fn test_compute_tensor_size_q4_0() {
        // 256 elements = 8 blocks * 18 bytes = 144
        assert_eq!(compute_tensor_size(&[256], GGMLType::Q4_0), 144);
    }

    #[test]
    fn test_compute_tensor_size_q4_k() {
        // 512 elements = 2 blocks * 144 bytes = 288
        assert_eq!(compute_tensor_size(&[512], GGMLType::Q4_K), 288);
    }

    #[test]
    fn test_compute_tensor_size_importance_quants() {
        for (ggml_type, expected_bytes) in [
            (GGMLType::IQ2_XXS, 66),
            (GGMLType::IQ2_XS, 74),
            (GGMLType::IQ3_XXS, 98),
            (GGMLType::IQ1_S, 50),
            (GGMLType::IQ3_S, 110),
            (GGMLType::IQ2_S, 82),
            (GGMLType::IQ4_XS, 136),
            (GGMLType::IQ1_M, 56),
        ] {
            assert_eq!(compute_tensor_size(&[256], ggml_type), expected_bytes);
        }
    }

    #[test]
    fn test_ggml_quant_params_f32() {
        assert_eq!(ggml_quant_params(GGMLType::F32), (1, 4));
    }

    #[test]
    fn test_ggml_quant_params_q6_k() {
        assert_eq!(ggml_quant_params(GGMLType::Q6_K), (256, 210));
    }

    #[test]
    fn test_map_tensors_f32() {
        use crate::gguf::parser::GGUFFile;
        use rnb_core::memory::mmap::MmapLoader;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // 최소 GGUF: F32 텐서 [8, 4] 1개
        let data = make_test_gguf_with_tensor();
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&data).unwrap();
        f.flush().unwrap();

        let mmap = MmapLoader::load_file_backed(f.path()).unwrap();
        let gguf = GGUFFile::parse(mmap.as_slice()).unwrap();
        let (tensors, _float_shapes, ggml_types, _file_offsets) = map_tensors(&gguf, mmap).unwrap();

        assert!(tensors.contains_key("token_embd.weight"));
        let t = &tensors["token_embd.weight"];
        assert_eq!(t.shape(), &[8, 4]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(ggml_types["token_embd.weight"], GGMLType::F32);
    }

    #[test]
    fn test_map_tensors_importance_quants() {
        use crate::gguf::parser::GGUFFile;
        use rnb_core::memory::mmap::MmapLoader;
        use std::io::Write;
        use tempfile::NamedTempFile;

        for ggml_type in [
            GGMLType::IQ2_XXS,
            GGMLType::IQ2_XS,
            GGMLType::IQ3_XXS,
            GGMLType::IQ1_S,
            GGMLType::IQ4_NL,
            GGMLType::IQ3_S,
            GGMLType::IQ2_S,
            GGMLType::IQ4_XS,
            GGMLType::IQ1_M,
            GGMLType::TQ1_0,
            GGMLType::TQ2_0,
        ] {
            let data_size = compute_tensor_size(&[256], ggml_type);
            let data = make_test_gguf_with_importance_quant_tensor(ggml_type, data_size);
            let mut file = NamedTempFile::new().unwrap();
            file.write_all(&data).unwrap();
            file.flush().unwrap();

            let mmap = MmapLoader::load_file_backed(file.path()).unwrap();
            let gguf = GGUFFile::parse(mmap.as_slice()).unwrap();
            let (tensors, float_shapes, ggml_types, _file_offsets) =
                map_tensors(&gguf, mmap).unwrap();
            let tensor = &tensors["test.weight"];
            assert_eq!(tensor.shape(), &[data_size]);
            assert_eq!(float_shapes["test.weight"], [256]);
            assert_eq!(tensor.dtype(), DType::U8);
            assert_eq!(ggml_types["test.weight"], ggml_type);
        }
    }

    #[test]
    fn test_compute_tensor_size_matches_data() {
        let size = compute_tensor_size(&[8, 4], GGMLType::F32);
        assert_eq!(size, 8 * 4 * 4); // 128 bytes
    }

    fn make_test_gguf_with_importance_quant_tensor(
        ggml_type: GGMLType,
        data_size: usize,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let value = "llama";
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value.as_bytes());
        let name = "test.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&256u64.to_le_bytes());
        buf.extend_from_slice(&(ggml_type as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.extend(std::iter::repeat_n(0, pad));
        buf.extend(std::iter::repeat_n(0, data_size));
        buf
    }

    /// 테스트용 최소 GGUF 바이너리 (F32 텐서 [8,4] 포함)
    fn make_test_gguf_with_tensor() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        // magic + version
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        // tensor_count=1, kv_count=1
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        // KV: "general.architecture" = "llama"
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // String type
        let val = "llama";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());
        // TensorInfo: name="token_embd.weight", shape=[8,4], F32, offset=0
        let name = "token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims=2
        buf.extend_from_slice(&4u64.to_le_bytes()); // innermost dim (stored reversed)
        buf.extend_from_slice(&8u64.to_le_bytes()); // outermost dim
        buf.extend_from_slice(&0u32.to_le_bytes()); // GGMLType::F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset=0
                                                    // align to 32
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.extend(std::iter::repeat(0u8).take(pad));
        // tensor data: 8*4*4 = 128 bytes
        buf.extend(std::iter::repeat(0u8).take(128));
        buf
    }
}
