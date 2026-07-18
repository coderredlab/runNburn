//! Drafter weight loader (`gemma4_assistant` GGUF → `Drafter`).
//!
//! mmap 한 번만 떠서 `Arc<Mmap>` 로 공유하고, quantized weight 는 zero-copy
//! 로 들고 다닌다. F32 norm/scalar 만 작은 `Vec<f32>` 로 본문에 복사한다.
//!
//! Spec: `docs/superpowers/specs/2026-05-13-gemma4-assistant-drafter-design.md`.

use super::types::{Drafter, DrafterLayer, TensorView};
use memmap2::Mmap;
use rnb_loader::arch::{extract_metadata, Architecture};
use rnb_loader::gguf::parser::GGUFFile;
use rnb_loader::gguf::types::{GGMLType, TensorInfo};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Drafter loader 가 reportable 한 error 타입.
#[derive(Debug, thiserror::Error)]
pub enum DrafterLoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
    #[error("metadata: {0}")]
    Metadata(String),
    #[error("missing assistant metadata block (general.architecture must be gemma4_assistant)")]
    MissingAssistant,
    #[error("tensor {name} missing")]
    MissingTensor { name: String },
    #[error("tensor {name}: expected shape {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("tensor {name}: expected dtype {expected:?}, got {got:?}")]
    DtypeMismatch {
        name: String,
        expected: GGMLType,
        got: GGMLType,
    },
    #[error("unsupported arch: {0:?}")]
    UnsupportedArch(Architecture),
    #[error("unsupported tensor dtype for byte-len calc: {0:?}")]
    UnsupportedDtype(GGMLType),
    #[error("tensor {name}: payload extends past mmap (offset={offset}, len={len}, mmap_len={mmap_len})")]
    OutOfBounds {
        name: String,
        offset: usize,
        len: usize,
        mmap_len: usize,
    },
}

/// `gemma4_assistant` GGUF 한 파일을 읽어 `Drafter` 로 반환한다.
pub fn load_drafter(path: &Path) -> Result<Drafter, DrafterLoadError> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mmap = Arc::new(mmap);

    let gguf = GGUFFile::parse(&mmap[..]).map_err(|e| DrafterLoadError::Parse(e.to_string()))?;
    let metadata =
        extract_metadata(&gguf.metadata).map_err(|e| DrafterLoadError::Metadata(e.to_string()))?;

    if metadata.architecture != Architecture::Gemma4Assistant {
        return Err(DrafterLoadError::UnsupportedArch(metadata.architecture));
    }
    let assistant = metadata
        .assistant
        .ok_or(DrafterLoadError::MissingAssistant)?;

    let block_count = metadata.num_layers;
    let hidden = metadata.hidden_size;
    let backbone_hidden = assistant.n_embd_backbone as usize;
    let n_heads = metadata.num_heads;
    let n_kv_heads = metadata.num_kv_heads;
    let head_dim_swa = assistant.key_length_swa as usize;
    let head_dim_full = assistant.key_length_full as usize;

    // Tensor lookup map (name → TensorInfo&) — single pass to avoid O(N²) scans.
    let tensors: HashMap<&str, &TensorInfo> = gguf
        .tensor_infos
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    let data_start = gguf.data_start;
    let mmap_len = mmap.len();

    // Helper: build a TensorView for a named tensor with expected shape + dtype.
    let take_tensor = |name: &str,
                       expected_shape: &[usize],
                       expected_dtype: GGMLType|
     -> Result<TensorView, DrafterLoadError> {
        let info = tensors
            .get(name)
            .copied()
            .ok_or_else(|| DrafterLoadError::MissingTensor {
                name: name.to_string(),
            })?;
        if info.shape != expected_shape {
            return Err(DrafterLoadError::ShapeMismatch {
                name: name.to_string(),
                expected: expected_shape.to_vec(),
                got: info.shape.clone(),
            });
        }
        if info.ggml_type != expected_dtype {
            return Err(DrafterLoadError::DtypeMismatch {
                name: name.to_string(),
                expected: expected_dtype,
                got: info.ggml_type,
            });
        }
        let offset = data_start
            .checked_add(info.offset as usize)
            .ok_or_else(|| DrafterLoadError::Parse(format!("offset overflow for {name}")))?;
        let len = tensor_byte_len(info)?;
        if offset
            .checked_add(len)
            .map(|end| end > mmap_len)
            .unwrap_or(true)
        {
            return Err(DrafterLoadError::OutOfBounds {
                name: name.to_string(),
                offset,
                len,
                mmap_len,
            });
        }
        Ok(TensorView {
            mmap: Arc::clone(&mmap),
            offset,
            len,
            ggml_type: info.ggml_type,
            shape: info.shape.clone(),
        })
    };

    // Same as `take_tensor` but accepts one of several allowed dtypes
    // (drafter `ffn_down` is Q4_K for layers 0-1, Q6_K for layers 2-3).
    let take_tensor_any = |name: &str,
                           expected_shape: &[usize],
                           allowed_dtypes: &[GGMLType]|
     -> Result<TensorView, DrafterLoadError> {
        let info = tensors
            .get(name)
            .copied()
            .ok_or_else(|| DrafterLoadError::MissingTensor {
                name: name.to_string(),
            })?;
        if info.shape != expected_shape {
            return Err(DrafterLoadError::ShapeMismatch {
                name: name.to_string(),
                expected: expected_shape.to_vec(),
                got: info.shape.clone(),
            });
        }
        if !allowed_dtypes.contains(&info.ggml_type) {
            return Err(DrafterLoadError::DtypeMismatch {
                name: name.to_string(),
                expected: allowed_dtypes[0],
                got: info.ggml_type,
            });
        }
        let offset = data_start
            .checked_add(info.offset as usize)
            .ok_or_else(|| DrafterLoadError::Parse(format!("offset overflow for {name}")))?;
        let len = tensor_byte_len(info)?;
        if offset
            .checked_add(len)
            .map(|end| end > mmap_len)
            .unwrap_or(true)
        {
            return Err(DrafterLoadError::OutOfBounds {
                name: name.to_string(),
                offset,
                len,
                mmap_len,
            });
        }
        Ok(TensorView {
            mmap: Arc::clone(&mmap),
            offset,
            len,
            ggml_type: info.ggml_type,
            shape: info.shape.clone(),
        })
    };

    // Helper: read an F32 tensor into an owned Vec<f32>. Uses byte-wise
    // `from_le_bytes` so we don't rely on the mmap pointer being 4-aligned.
    let take_f32 = |name: &str, expected_shape: &[usize]| -> Result<Vec<f32>, DrafterLoadError> {
        let view = take_tensor(name, expected_shape, GGMLType::F32)?;
        let bytes = view.as_bytes();
        let count: usize = expected_shape.iter().product();
        if bytes.len() != count * 4 {
            return Err(DrafterLoadError::Parse(format!(
                "{name}: F32 byte len {} != {} * 4",
                bytes.len(),
                count
            )));
        }
        let mut out = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    };

    // --- Global tensors -----------------------------------------------------

    let token_embd = take_tensor(
        "token_embd.weight",
        &[metadata.vocab_size, hidden],
        GGMLType::Q6_K,
    )?;
    let output_norm = take_f32("output_norm.weight", &[hidden])?;
    // rope_freqs.weight 는 inv freq table: 길이 = key_length_full / 2 (=256).
    let rope_freqs = take_f32("rope_freqs.weight", &[head_dim_full / 2])?;
    let pre_projection = take_tensor(
        "mtp.pre_projection.weight",
        // spec §"Stage B" — pre_projection input = 2 × backbone_hidden
        // (target current+previous hidden concat 후보; 정확한 의미는 Task 4).
        &[hidden, 2 * backbone_hidden],
        GGMLType::Q4_K,
    )?;
    let post_projection = take_tensor(
        "mtp.post_projection.weight",
        &[backbone_hidden, hidden],
        GGMLType::Q4_K,
    )?;
    let centroids = take_tensor(
        "mtp.centroids.weight",
        &[assistant.n_centroids as usize, hidden],
        GGMLType::Q4_K,
    )?;

    // `mtp.token_ordering.weight` I32 [vocab_size] — transformers source
    // 의 explicit permutation buffer. token_id → cluster slot. Q4_K_M GGUF
    // 에 함께 박혀있다. byte-wise i32→u32 로 읽어들임 (token id 는 항상
    // non-negative 이고 i32/u32 byte layout 이 동일).
    let token_ordering_view = take_tensor(
        "mtp.token_ordering.weight",
        &[metadata.vocab_size],
        GGMLType::I32,
    )?;
    let token_ordering = {
        let bytes = token_ordering_view.as_bytes();
        if bytes.len() != metadata.vocab_size * 4 {
            return Err(DrafterLoadError::Parse(format!(
                "mtp.token_ordering.weight: byte len {} != {} * 4",
                bytes.len(),
                metadata.vocab_size
            )));
        }
        let mut out = Vec::with_capacity(metadata.vocab_size);
        for chunk in bytes.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out
    };

    // --- Per-layer tensors --------------------------------------------------

    let sliding_pattern = &assistant.sliding_window_pattern;
    if sliding_pattern.len() != block_count {
        return Err(DrafterLoadError::Metadata(format!(
            "sliding_window_pattern len {} != block_count {block_count}",
            sliding_pattern.len()
        )));
    }

    let ffn_dim = metadata.intermediate_size; // 2048

    let mut layers = Vec::with_capacity(block_count);
    for layer_idx in 0..block_count {
        let is_swa = sliding_pattern[layer_idx];
        // SWA vs full attention layout. SWA uses head_dim=256 (key_length_swa)
        // so attn_q output rows = n_heads * 256 = 1024; full uses 512 → 2048.
        let head_dim = if is_swa { head_dim_swa } else { head_dim_full };
        let q_out = n_heads * head_dim;
        let q_norm_len = head_dim;

        let prefix = format!("blk.{layer_idx}");
        let attn_norm = take_f32(&format!("{prefix}.attn_norm.weight"), &[hidden])?;
        let attn_q_norm = take_f32(&format!("{prefix}.attn_q_norm.weight"), &[q_norm_len])?;
        let ffn_norm = take_f32(&format!("{prefix}.ffn_norm.weight"), &[hidden])?;
        let post_attention_norm =
            take_f32(&format!("{prefix}.post_attention_norm.weight"), &[hidden])?;
        let post_ffw_norm = take_f32(&format!("{prefix}.post_ffw_norm.weight"), &[hidden])?;
        let layer_output_scale_vec =
            take_f32(&format!("{prefix}.layer_output_scale.weight"), &[1])?;
        let layer_output_scale = layer_output_scale_vec[0];

        let attn_q = take_tensor(
            &format!("{prefix}.attn_q.weight"),
            &[q_out, hidden],
            GGMLType::Q4_K,
        )?;
        let attn_output = take_tensor(
            &format!("{prefix}.attn_output.weight"),
            &[hidden, q_out],
            GGMLType::Q4_K,
        )?;
        let ffn_gate = take_tensor(
            &format!("{prefix}.ffn_gate.weight"),
            &[ffn_dim, hidden],
            GGMLType::Q4_K,
        )?;
        let ffn_up = take_tensor(
            &format!("{prefix}.ffn_up.weight"),
            &[ffn_dim, hidden],
            GGMLType::Q4_K,
        )?;
        // ffn_down: drafter mixes Q4_K (layers 0-1) and Q6_K (layers 2-3).
        // We accept both, then the forward kernel dispatches on `ggml_type`.
        let ffn_down = take_tensor_any(
            &format!("{prefix}.ffn_down.weight"),
            &[hidden, ffn_dim],
            &[GGMLType::Q4_K, GGMLType::Q6_K],
        )?;

        layers.push(DrafterLayer {
            layer_idx,
            is_sliding_window: is_swa,
            head_dim,
            n_heads,
            n_kv_heads,
            attn_norm,
            attn_q_norm,
            ffn_norm,
            post_attention_norm,
            post_ffw_norm,
            layer_output_scale,
            attn_q,
            attn_output,
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    Ok(Drafter {
        block_count,
        hidden,
        backbone_hidden,
        n_centroids: assistant.n_centroids,
        centroid_top_k: assistant.centroid_top_k,
        sliding_window: assistant.sliding_window as usize,
        token_embd,
        output_norm,
        rope_freqs,
        pre_projection,
        post_projection,
        centroids,
        token_ordering,
        layers,
    })
}

/// GGUF tensor 한 개의 원시 byte length 를 dtype + shape 로부터 계산.
///
/// K-quant 슈퍼블록 크기 (Q4_K=144B / 256 elem, Q6_K=210B / 256 elem) 는
/// llama.cpp ggml 표준값. 다른 dtype 이 들어오면 명시적으로 reject.
fn tensor_byte_len(t: &TensorInfo) -> Result<usize, DrafterLoadError> {
    let elem_count: usize = t.shape.iter().product();
    match t.ggml_type {
        GGMLType::F32 => Ok(elem_count * 4),
        GGMLType::F16 | GGMLType::BF16 => Ok(elem_count * 2),
        GGMLType::I32 => Ok(elem_count * 4),
        // K-quant super-block = 256 elements; Q4_K=144B, Q6_K=210B per block.
        GGMLType::Q4_K => {
            if elem_count % 256 != 0 {
                return Err(DrafterLoadError::Parse(format!(
                    "{}: Q4_K elem_count {elem_count} not multiple of 256",
                    t.name
                )));
            }
            Ok((elem_count / 256) * 144)
        }
        GGMLType::Q6_K => {
            if elem_count % 256 != 0 {
                return Err(DrafterLoadError::Parse(format!(
                    "{}: Q6_K elem_count {elem_count} not multiple of 256",
                    t.name
                )));
            }
            Ok((elem_count / 256) * 210)
        }
        other => Err(DrafterLoadError::UnsupportedDtype(other)),
    }
}
