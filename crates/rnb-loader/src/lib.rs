pub mod arch;
pub mod cache;
pub mod convert;
pub mod error;
pub mod gguf;
mod mtp_sidecar;
pub mod packed;
pub mod rnb_file;
pub mod rnb_moe_reader;
pub mod sidecar_v3;

pub use arch::{Architecture, ModelLayerKind, ModelMetadata, MtpLayerTensors, MtpMetadata};
pub use error::LoaderError;
pub use gguf::types::GGMLType;
pub use rnb_core::tensor::QuantType;
// Re-export the residency view trait so engine consumers can pull it through
// `rnb-loader` (the crate that already owns the file-format reader and is
// the natural consumer-side handle for MoE expert byte slices). The trait
// itself lives in `rnb-memory`, which is the long-term owner of memory tier
// abstractions.
pub use rnb_memory::moe_residency::MoeExpertResidencyView;

use std::collections::HashMap;
use std::path::Path;

use rnb_core::ir::graph::Graph;
use rnb_core::memory::mmap::MmapLoader;
use rnb_core::tensor::tensor::Tensor;

use arch::{
    build_graph, collect_mtp_layer_tensors, detect_architecture, extract_metadata,
    infer_nemotron_layer_kinds_from_tensor_names,
};
use gguf::parser::GGUFFile;

/// GGUF 메타데이터에서 파싱한 토크나이저 데이터
#[derive(Debug, Clone)]
pub struct TokenizerData {
    /// vocab 크기 (tokens.len()과 동일)
    pub vocab_size: usize,
    /// token strings (index = token id)
    pub tokens: Vec<String>,
    /// token scores (SentencePiece BPE 우선순위)
    pub scores: Vec<f32>,
    /// BPE merge rules ("token1 token2" 형식)
    pub merges: Vec<String>,
    /// BOS token id
    pub bos_id: u32,
    /// EOS token id
    pub eos_id: u32,
    /// 토크나이저 모델 타입 ("llama" = SentencePiece, "gpt2" = GPT-2 BPE)
    pub model: String,
    /// GGUF Jinja chat template used to serialize role/content messages.
    pub chat_template: Option<String>,
    pub add_bos_token: bool,
    pub add_space_prefix: bool,
}

impl TokenizerData {
    /// 테스트/placeholder용: vocab_size만 있고 실제 토큰 목록은 없는 빈 데이터
    pub fn placeholder(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            tokens: Vec::new(),
            scores: Vec::new(),
            merges: Vec::new(),
            bos_id: 1,
            eos_id: 2,
            model: String::new(),
            chat_template: None,
            add_bos_token: true,
            add_space_prefix: true,
        }
    }
}

/// GGUF에서 로드된 모델의 완성 표현
pub struct LoadedModel {
    pub graph: Graph,
    pub weights: HashMap<String, Tensor>,
    pub metadata: ModelMetadata,
    /// 양자화 ���서의 원래 float shape: 이름 → [row, col, ...]
    /// weights map��서 텐서가 [byte_count] 1D로 저장된 경우의 원�� shape.
    pub float_shapes: HashMap<String, Vec<usize>>,
    /// 각 텐서�� 원래 GGUF 양자화 타입
    pub tensor_ggml_types: HashMap<String, GGMLType>,
    /// 각 텐서의 GGUF 파일 내 절대 byte offset (pread용)
    pub tensor_file_offsets: HashMap<String, usize>,
    /// GGUF NextN/MTP head weight names grouped by prediction layer.
    pub mtp_tensors: Vec<MtpLayerTensors>,
}

/// GGUF 파일을 로드하여 `LoadedModel`을 반환한다.
///
/// 내부적으로:
/// 1. mmap으로 파일을 열어 GGUF 바이너리 파싱
/// 2. 메타데이터에서 아키텍처/하이퍼파라미터 추출
/// 3. 텐서를 zero-copy mmap 뷰로 매핑
/// 4. 아키텍처에 맞는 IR Graph 빌드
pub fn load_model(path: &Path) -> Result<LoadedModel, LoaderError> {
    let mapped = gguf::sharded::load_mapped_gguf(path)?;
    let mut metadata = extract_metadata(&mapped.metadata)?;
    let mut weights = mapped.weights;
    let mut float_shapes = mapped.float_shapes;
    let mut tensor_ggml_types = mapped.tensor_ggml_types;
    let tensor_file_offsets = mapped.tensor_file_offsets;
    if metadata.architecture == Architecture::NemotronHMoE {
        let tensor_names = weights.keys().map(String::as_str).collect::<Vec<_>>();
        metadata.layer_kinds =
            infer_nemotron_layer_kinds_from_tensor_names(tensor_names, metadata.num_layers)?;
    }
    if metadata.mtp.is_none() && metadata.architecture == Architecture::Qwen35 {
        mtp_sidecar::attach_adjacent_qwen35_mtp1_sidecar(
            path,
            &mut metadata,
            &mut weights,
            &mut float_shapes,
            &mut tensor_ggml_types,
        )?;
    }
    let mtp_tensors = collect_mtp_layer_tensors(weights.keys().map(String::as_str), &metadata)?;
    let graph = build_graph(&metadata)?;
    Ok(LoadedModel {
        graph,
        weights,
        metadata,
        float_shapes,
        tensor_ggml_types,
        tensor_file_offsets,
        mtp_tensors,
    })
}

pub fn detect_model_architecture(path: &Path) -> Result<Architecture, LoaderError> {
    let mmap = MmapLoader::load(path)?;
    let gguf = GGUFFile::parse(&mmap[..])?;
    detect_architecture(&gguf.metadata)
}

pub fn model_ir_from_metadata(metadata: &ModelMetadata) -> rnb_model_ir::ModelGraph {
    let mut graph = rnb_model_ir::ModelGraph::new(rnb_model_ir::ModelKind::DecoderOnly);
    for layer_idx in 0..metadata.num_layers {
        let kind = match metadata.layer_kinds.get(layer_idx).copied() {
            Some(ModelLayerKind::Recurrent) => rnb_model_ir::LayerKind::Gdn,
            Some(ModelLayerKind::Attention | ModelLayerKind::MoE) | None => {
                rnb_model_ir::LayerKind::Attention
            }
        };
        graph.push_layer(rnb_model_ir::LayerSpec::new(
            rnb_model_ir::LayerId(layer_idx),
            kind,
        ));
    }
    graph
}

#[cfg(test)]
mod model_ir_tests {
    use super::*;

    fn metadata_with_attention_interval(interval: usize) -> ModelMetadata {
        ModelMetadata {
            architecture: Architecture::Qwen35MoE,
            vocab_size: 1024,
            hidden_size: 128,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 32,
            intermediate_size: 256,
            max_seq_len: 1024,
            rope_theta: 1_000_000.0,
            rope_theta_swa: 1_000_000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-6,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 0.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: Vec::new(),
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            embedding_length_per_layer_input: 0,
            expert_count: 0,
            expert_used_count: 0,
            expert_shared_count: 0,
            leading_dense_block_count: 0,
            expert_gating_func: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            expert_feed_forward_length: 0,
            head_count_kv_per_layer: None,
            tokenizer: TokenizerData::placeholder(1024),
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: interval,
            layer_kinds: (0..4)
                .map(|layer_idx| {
                    if interval > 0 && layer_idx % interval != interval.saturating_sub(1) {
                        ModelLayerKind::Recurrent
                    } else {
                        ModelLayerKind::Attention
                    }
                })
                .collect(),
            mtp: None,
            assistant: None,
        }
    }

    #[test]
    fn metadata_to_model_ir_marks_gdn_layers_between_attention_layers() {
        let graph = model_ir_from_metadata(&metadata_with_attention_interval(2));
        let kinds = graph
            .layers()
            .iter()
            .map(|layer| layer.kind)
            .collect::<Vec<_>>();

        assert_eq!(
            kinds,
            vec![
                rnb_model_ir::LayerKind::Gdn,
                rnb_model_ir::LayerKind::Attention,
                rnb_model_ir::LayerKind::Gdn,
                rnb_model_ir::LayerKind::Attention,
            ]
        );
    }
}

#[cfg(test)]
mod mtp_sidecar_tests {
    use super::*;
    use crate::gguf::types::GGMLType;
    use rnb_core::tensor::DType;
    use rnb_core::tensor::Tensor;
    use std::collections::HashMap;

    fn write_tensor_record(
        header: &mut Vec<u8>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[u32],
        data: &[u8],
    ) {
        header.extend_from_slice(&(name.len() as u32).to_le_bytes());
        header.extend_from_slice(name.as_bytes());
        header.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape {
            header.extend_from_slice(&dim.to_le_bytes());
        }
        header.extend_from_slice(&(GGMLType::F16 as u32).to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        header.extend_from_slice(&(data.len() as u64).to_le_bytes());
        payload.extend_from_slice(data);
    }

    fn f16_bytes(bits: &[u16]) -> Vec<u8> {
        bits.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn make_mtp1_sidecar() -> Vec<u8> {
        let entries = [
            ("mtp.fc.weight", vec![2, 4], vec![1u8; 16]),
            (
                "mtp.layers.0.input_layernorm.weight",
                vec![2],
                f16_bytes(&[0x3800, 0x3400]),
            ),
            (
                "mtp.layers.0.mlp.down_proj.weight",
                vec![2, 3],
                vec![3u8; 12],
            ),
            (
                "mtp.layers.0.mlp.gate_proj.weight",
                vec![3, 2],
                vec![4u8; 12],
            ),
            ("mtp.layers.0.mlp.up_proj.weight", vec![3, 2], vec![5u8; 12]),
            (
                "mtp.layers.0.post_attention_layernorm.weight",
                vec![2],
                f16_bytes(&[0x3000, 0x2c00]),
            ),
            (
                "mtp.layers.0.self_attn.k_norm.weight",
                vec![1],
                f16_bytes(&[0x3800]),
            ),
            (
                "mtp.layers.0.self_attn.k_proj.weight",
                vec![1, 2],
                vec![8u8; 4],
            ),
            (
                "mtp.layers.0.self_attn.o_proj.weight",
                vec![2, 2],
                vec![9u8; 8],
            ),
            (
                "mtp.layers.0.self_attn.q_norm.weight",
                vec![1],
                f16_bytes(&[0x3800]),
            ),
            (
                "mtp.layers.0.self_attn.q_proj.weight",
                vec![4, 2],
                vec![11u8; 16],
            ),
            (
                "mtp.layers.0.self_attn.v_proj.weight",
                vec![1, 2],
                vec![12u8; 4],
            ),
            ("mtp.norm.weight", vec![2], f16_bytes(&[0x3800, 0x3400])),
            (
                "mtp.pre_fc_norm_embedding.weight",
                vec![2],
                f16_bytes(&[0x3800, 0x3400]),
            ),
            (
                "mtp.pre_fc_norm_hidden.weight",
                vec![2],
                f16_bytes(&[0x3800, 0x3400]),
            ),
        ];

        let mut header = Vec::new();
        let mut payload = Vec::new();
        header.extend_from_slice(b"MTP1");
        header.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (name, shape, data) in &entries {
            write_tensor_record(&mut header, &mut payload, name, shape, data);
        }

        let header_len = header.len();
        let mut cursor = 8usize;
        let mut data_offset = header_len as u64;
        for (name, shape, data) in &entries {
            cursor += 4 + name.len() + 4 + shape.len() * 4 + 4;
            header[cursor..cursor + 8].copy_from_slice(&data_offset.to_le_bytes());
            cursor += 8;
            cursor += 8;
            data_offset += data.len() as u64;
        }

        header.extend_from_slice(&payload);
        header
    }

    #[test]
    fn mtp1_sidecar_parser_preserves_tensor_bytes_and_shapes() {
        let bytes = make_mtp1_sidecar();

        let tensors = crate::mtp_sidecar::parse_mtp1_sidecar_bytes(&bytes).unwrap();

        let fc = tensors.iter().find(|t| t.name == "mtp.fc.weight").unwrap();
        assert_eq!(fc.shape, vec![2, 4]);
        assert_eq!(fc.ggml_type, GGMLType::F16);
        assert_eq!(fc.data, vec![1u8; 16]);
    }

    #[test]
    fn qwen35_mtp1_sidecar_injects_runtime_weight_names() {
        let bytes = make_mtp1_sidecar();
        let mut metadata = ModelMetadata {
            architecture: Architecture::Qwen35,
            vocab_size: 1024,
            hidden_size: 2,
            num_layers: 24,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            intermediate_size: 3,
            max_seq_len: 128,
            rope_theta: 1_000_000.0,
            rope_theta_swa: 1_000_000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-6,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 1.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: Vec::new(),
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            embedding_length_per_layer_input: 0,
            expert_count: 0,
            expert_used_count: 0,
            expert_shared_count: 0,
            leading_dense_block_count: 0,
            expert_gating_func: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            expert_feed_forward_length: 0,
            head_count_kv_per_layer: None,
            tokenizer: TokenizerData::placeholder(1024),
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: 4,
            layer_kinds: Vec::new(),
            mtp: None,
            assistant: None,
        };
        metadata.architecture = Architecture::Qwen35;
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        let mut float_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let mut tensor_ggml_types: HashMap<String, GGMLType> = HashMap::new();

        crate::mtp_sidecar::inject_qwen35_mtp1_sidecar_bytes(
            &bytes,
            &mut metadata,
            &mut weights,
            &mut float_shapes,
            &mut tensor_ggml_types,
        )
        .unwrap();

        let mtp = metadata.mtp.unwrap();
        assert_eq!(mtp.trunk_layers, 24);
        assert_eq!(mtp.first_mtp_layer, 24);
        assert_eq!(mtp.nextn_predict_layers, 1);
        assert_eq!(float_shapes["blk.24.nextn.eh_proj.weight"], vec![2, 4]);
        assert_eq!(float_shapes["blk.24.attn_q.weight"], vec![4, 2]);
        assert_eq!(
            tensor_ggml_types["blk.24.attn_output.weight"],
            GGMLType::F16
        );
        assert_eq!(weights["blk.24.nextn.eh_proj.weight"].dtype(), DType::U8);
        assert_eq!(
            weights["blk.24.nextn.eh_proj.weight"].as_bytes().unwrap(),
            &[1u8; 16]
        );
        let enorm = weights["blk.24.nextn.enorm.weight"]
            .as_bytes()
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>();
        assert_eq!(weights["blk.24.nextn.enorm.weight"].dtype(), DType::F32);
        assert_eq!(enorm, vec![1.5, 1.25]);
        assert!(weights.contains_key("blk.24.post_attention_norm.weight"));
    }

    #[test]
    fn qwen35_mtp1_sidecar_can_materialize_matmul_weights_as_f32() {
        let bytes = make_mtp1_sidecar();
        let mut metadata = ModelMetadata {
            architecture: Architecture::Qwen35,
            vocab_size: 1024,
            hidden_size: 2,
            num_layers: 24,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            intermediate_size: 3,
            max_seq_len: 128,
            rope_theta: 1_000_000.0,
            rope_theta_swa: 1_000_000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-6,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 1.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: Vec::new(),
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            embedding_length_per_layer_input: 0,
            expert_count: 0,
            expert_used_count: 0,
            expert_shared_count: 0,
            leading_dense_block_count: 0,
            expert_gating_func: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            expert_feed_forward_length: 0,
            head_count_kv_per_layer: None,
            tokenizer: TokenizerData::placeholder(1024),
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: 4,
            layer_kinds: Vec::new(),
            mtp: None,
            assistant: None,
        };
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        let mut float_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let mut tensor_ggml_types: HashMap<String, GGMLType> = HashMap::new();

        crate::mtp_sidecar::inject_qwen35_mtp1_sidecar_bytes_with_options(
            &bytes,
            &mut metadata,
            &mut weights,
            &mut float_shapes,
            &mut tensor_ggml_types,
            crate::mtp_sidecar::MtpSidecarLoadOptions::with_materialize_f16_matmul_as_f32(true),
        )
        .unwrap();

        assert_eq!(weights["blk.24.nextn.eh_proj.weight"].dtype(), DType::F32);
        assert!(!float_shapes.contains_key("blk.24.nextn.eh_proj.weight"));
        assert_eq!(
            tensor_ggml_types["blk.24.nextn.eh_proj.weight"],
            GGMLType::F32
        );
    }
}
