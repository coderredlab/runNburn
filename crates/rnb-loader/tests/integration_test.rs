//! 통합 테스트: 합성 GGUF 바이너리로 전체 load_model 파이프라인 검증

use rnb_loader::{load_model, Architecture, LoaderError};
use std::io::Write;
use tempfile::NamedTempFile;

/// 최소 LLaMA GGUF 파일 생성 (token_embd.weight F32 [32, 64] 텐서 1개)
fn write_mini_llama_gguf() -> NamedTempFile {
    let mut buf: Vec<u8> = Vec::new();

    // Magic + version 3
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());

    // tensor_count=1, kv_count=9
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&9u64.to_le_bytes());

    // --- KV pairs ---
    // KV String helper
    let kv_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // value_type = String
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
    };
    let kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // value_type = U32
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let kv_f32 = |buf: &mut Vec<u8>, k: &str, v: f32| {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes()); // value_type = F32
        buf.extend_from_slice(&v.to_le_bytes());
    };

    kv_str(&mut buf, "general.architecture", "llama");
    kv_u32(&mut buf, "llama.embedding_length", 64);
    kv_u32(&mut buf, "llama.block_count", 2);
    kv_u32(&mut buf, "llama.attention.head_count", 4);
    kv_u32(&mut buf, "llama.attention.head_count_kv", 4);
    kv_u32(&mut buf, "llama.feed_forward_length", 128);
    kv_u32(&mut buf, "llama.context_length", 512);
    kv_f32(&mut buf, "llama.rope.freq_base", 10000.0);
    kv_f32(&mut buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);

    // --- TensorInfo: "token_embd.weight" F32 shape=[32, 64] ---
    // name
    let name = "token_embd.weight";
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    // n_dims = 2, dims stored innermost-first → [64, 32] → reversed to [32, 64]
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&64u64.to_le_bytes()); // innermost
    buf.extend_from_slice(&32u64.to_le_bytes()); // outermost
                                                 // GGMLType::F32 = 0
    buf.extend_from_slice(&0u32.to_le_bytes());
    // offset = 0
    buf.extend_from_slice(&0u64.to_le_bytes());

    // Pad to alignment=32
    let pad = (32 - (buf.len() % 32)) % 32;
    buf.extend(std::iter::repeat(0u8).take(pad));

    // Tensor data: 32*64*4 = 8192 bytes of zeros
    buf.extend(std::iter::repeat(0u8).take(32 * 64 * 4));

    let mut f = NamedTempFile::new().unwrap();
    f.write_all(&buf).unwrap();
    f.flush().unwrap();
    f
}

fn write_mini_qwen35moe_mtp_gguf() -> NamedTempFile {
    let mut buf: Vec<u8> = Vec::new();

    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());

    let tensor_specs = [
        ("blk.40.nextn.eh_proj.weight", vec![16u64, 8u64], 8 * 16 * 4),
        ("blk.40.nextn.enorm.weight", vec![8u64], 8 * 4),
        ("blk.40.nextn.hnorm.weight", vec![8u64], 8 * 4),
        ("blk.40.nextn.shared_head_norm.weight", vec![8u64], 8 * 4),
    ];

    buf.extend_from_slice(&(tensor_specs.len() as u64).to_le_bytes());
    buf.extend_from_slice(&12u64.to_le_bytes());

    let kv_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
    };
    let kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let kv_f32 = |buf: &mut Vec<u8>, k: &str, v: f32| {
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    };

    kv_str(&mut buf, "general.architecture", "qwen35moe");
    kv_u32(&mut buf, "qwen35moe.embedding_length", 8);
    kv_u32(&mut buf, "qwen35moe.block_count", 41);
    kv_u32(&mut buf, "qwen35moe.nextn_predict_layers", 1);
    kv_u32(&mut buf, "qwen35moe.attention.head_count", 2);
    kv_u32(&mut buf, "qwen35moe.attention.head_count_kv", 1);
    kv_u32(&mut buf, "qwen35moe.expert_feed_forward_length", 16);
    kv_u32(&mut buf, "qwen35moe.context_length", 512);
    kv_f32(&mut buf, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6);
    kv_u32(&mut buf, "qwen35moe.expert_count", 4);
    kv_u32(&mut buf, "qwen35moe.expert_used_count", 2);
    kv_u32(&mut buf, "qwen35moe.full_attention_interval", 4);

    let mut offset = 0u64;
    for (name, dims, bytes) in &tensor_specs {
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dim in dims {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        offset += *bytes as u64;
    }

    let pad = (32 - (buf.len() % 32)) % 32;
    buf.extend(std::iter::repeat(0u8).take(pad));
    for (_, _, bytes) in &tensor_specs {
        buf.extend(std::iter::repeat(0u8).take(*bytes));
    }

    let mut f = NamedTempFile::new().unwrap();
    f.write_all(&buf).unwrap();
    f.flush().unwrap();
    f
}

#[test]
fn test_load_model_llama_end_to_end() {
    let f = write_mini_llama_gguf();
    let model = load_model(f.path()).expect("load_model should succeed");

    assert_eq!(model.metadata.architecture, Architecture::LLaMA);
    assert_eq!(model.metadata.num_layers, 2);
    assert_eq!(model.metadata.hidden_size, 64);
    assert!(model.weights.contains_key("token_embd.weight"));

    let t = &model.weights["token_embd.weight"];
    assert_eq!(t.shape(), &[32, 64]);

    // 그래프 유효성 검증
    assert!(model.graph.validate().is_ok());
    assert!(model.graph.topological_order().is_ok());
    assert_eq!(model.graph.output_nodes().len(), 1);
}

#[test]
fn test_load_model_collects_qwen35moe_mtp_tensors() {
    let f = write_mini_qwen35moe_mtp_gguf();
    let model = load_model(f.path()).expect("load_model should succeed");

    let mtp = model.metadata.mtp.as_ref().expect("MTP metadata");
    assert_eq!(mtp.first_mtp_layer, 40);
    assert_eq!(mtp.nextn_predict_layers, 1);
    assert_eq!(model.mtp_tensors.len(), 1);

    let layer = &model.mtp_tensors[0];
    assert_eq!(layer.layer_index, 40);
    assert_eq!(layer.eh_proj_weight, "blk.40.nextn.eh_proj.weight");
    assert_eq!(layer.enorm_weight, "blk.40.nextn.enorm.weight");
    assert_eq!(layer.hnorm_weight, "blk.40.nextn.hnorm.weight");
    assert_eq!(
        layer.shared_head_norm_weight,
        "blk.40.nextn.shared_head_norm.weight"
    );
}

#[test]
fn test_load_model_invalid_path() {
    let result = load_model(std::path::Path::new("/no/such/file.gguf"));
    // MmapLoader::load는 RnbError::IoError를 반환하며 LoaderError::CoreError로 감싸짐
    assert!(result.is_err());
    match result {
        Err(LoaderError::IoError(_)) | Err(LoaderError::CoreError(_)) => {}
        Err(other) => panic!("expected IoError or CoreError, got: {other}"),
        Ok(_) => panic!("expected error for nonexistent path"),
    }
}

#[test]
fn test_load_model_invalid_magic() {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(b"XXXX\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
        .unwrap();
    f.flush().unwrap();
    let result = load_model(f.path());
    assert!(matches!(result, Err(LoaderError::InvalidMagic)));
}
