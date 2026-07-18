//! Drafter weight types.
//!
//! `TensorView` 는 GGUF mmap 안의 quantized weight 한 덩어리를 가리키는
//! zero-copy view. `Drafter` / `DrafterLayer` 는 51 tensor 의 전체 layout 을
//! 모은 owning aggregate 다.
//!
//! - F32 norm/scalar 텐서는 본문에 owning `Vec<f32>` 로 복사 (작은 사이즈 +
//!   forward 에서 자주 접근).
//! - Quantized weight (Q4_K / Q6_K) 와 token_embedding 은 mmap slice 를
//!   `Arc<memmap2::Mmap>` + offset/len 으로만 들고 다님.
//!
//! Spec: `docs/superpowers/specs/2026-05-13-gemma4-assistant-drafter-design.md`
//! "Tensor 목록" 섹션 참조.

use memmap2::Mmap;
use rnb_cpu::quantize::blocks::BlockQ6_K;
use rnb_cpu::quantize::dequant::dequantize_q6_k;
use rnb_loader::gguf::types::GGMLType;
use std::mem::size_of;
use std::sync::Arc;

/// VQ codebook (`mtp.centroids.weight`): `[n_centroids, centroid_dim]` Q4_K.
///
/// `pre_projection` 출력에서 top-K 가까운 centroid 를 골라 backbone 으로
/// 풀어내는 quantized intermediate. forward 시 `top_k` 만 활성화된다.
#[derive(Debug)]
pub struct VQCodebook {
    pub weight_bytes: Arc<Mmap>,
    pub offset: usize,
    pub len: usize,
    pub n_centroids: u32,
    pub centroid_dim: u32,
    pub top_k: u32,
}

/// Zero-copy quantized tensor view (GGUF mmap 안의 한 슬라이스).
///
/// `shape` 는 row-major (GGUF parser 가 reverse 해서 정규화한 형태).
#[derive(Debug, Clone)]
pub struct TensorView {
    pub mmap: Arc<Mmap>,
    pub offset: usize,
    pub len: usize,
    pub ggml_type: GGMLType,
    pub shape: Vec<usize>,
}

impl TensorView {
    /// Raw bytes view. Mmap 은 `Arc` 가 살아있는 동안 유효.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[self.offset..self.offset + self.len]
    }
}

/// Drafter 의 단일 transformer block. attention 은 SWA (sliding-window) /
/// full 중 하나로 layout 이 갈린다 (`is_sliding_window`).
///
/// drafter 는 `attn_k` / `attn_v` 가 없다 (target 모델과 KV cache 공유).
#[derive(Debug)]
pub struct DrafterLayer {
    pub layer_idx: usize,
    pub is_sliding_window: bool,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,

    // F32 norms / scalars (owned, small).
    pub attn_norm: Vec<f32>,
    pub attn_q_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    pub layer_output_scale: f32,

    // Quantized weights (zero-copy).
    pub attn_q: TensorView,
    pub attn_output: TensorView,
    pub ffn_gate: TensorView,
    pub ffn_up: TensorView,
    pub ffn_down: TensorView,
}

/// 전체 drafter — global tensor + layer 모음.
#[derive(Debug)]
pub struct Drafter {
    pub block_count: usize,
    pub hidden: usize,
    pub backbone_hidden: usize,
    pub n_centroids: u32,
    pub centroid_top_k: u32,
    /// SWA window size in tokens (`attention.sliding_window` = 512).
    /// drafter cross-attention sliding window 크기로 사용한다.
    pub sliding_window: usize,

    pub token_embd: TensorView,
    pub output_norm: Vec<f32>,
    pub rope_freqs: Vec<f32>,
    pub pre_projection: TensorView,
    pub post_projection: TensorView,
    pub centroids: TensorView,

    /// `mtp.token_ordering.weight` I32 [vocab_size]. transformers source
    /// (`Gemma4AssistantMaskedEmbedder.forward`) 에 따르면 token_id → cluster
    /// slot position 의 explicit permutation. cluster `c` 의 token 집합 =
    /// `token_ordering[c*vocab_per_centroid .. (c+1)*vocab_per_centroid]`,
    /// `vocab_per_centroid = vocab_size / n_centroids = 128`.
    pub token_ordering: Vec<u32>,

    pub layers: Vec<DrafterLayer>,
}

impl Drafter {
    /// Drafter 의 `token_embd.weight` (Q6_K [vocab, hidden]) 에서 한 token row
    /// 를 dequant 해서 `Vec<f32>` (len = hidden = 256) 로 반환.
    ///
    /// transformers `Gemma4AssistantMaskedEmbedder.forward` 는 lm_head 가
    /// `model.embed_tokens` 와 tied 라고 정의한다. 즉 vocab_logits = `x · drafter.token_embd[tok]`
    /// (256-dim dot product) 이다. target Engine 의 2560-dim token_embd 와
    /// 다른 별개 tensor.
    ///
    /// hidden = 256 이라 row 당 정확히 K-quant block 1개 (256 elem = 210 byte).
    pub fn token_embd_row(&self, token_id: u32) -> Vec<f32> {
        assert_eq!(
            self.token_embd.ggml_type,
            GGMLType::Q6_K,
            "token_embd dtype must be Q6_K"
        );
        assert_eq!(
            self.token_embd.shape.len(),
            2,
            "token_embd shape rank != 2: {:?}",
            self.token_embd.shape
        );
        let cols = self.token_embd.shape[1];
        assert_eq!(
            cols, self.hidden,
            "token_embd cols {} != drafter.hidden {}",
            cols, self.hidden
        );
        assert!(
            cols % 256 == 0,
            "token_embd cols {} not multiple of K-quant block size 256",
            cols
        );
        let blocks_per_row = cols / 256;
        let bytes_per_row = blocks_per_row * size_of::<BlockQ6_K>();
        let row_start = (token_id as usize) * bytes_per_row;
        let row_end = row_start + bytes_per_row;
        let view_bytes = self.token_embd.as_bytes();
        assert!(
            row_end <= view_bytes.len(),
            "token_embd row {token_id}: end {row_end} > view len {}",
            view_bytes.len()
        );
        let row_bytes = &view_bytes[row_start..row_end];

        let mut out = vec![0.0f32; cols];
        let mut tmp = [0.0f32; 256];
        for bi in 0..blocks_per_row {
            let off = bi * size_of::<BlockQ6_K>();
            let block: BlockQ6_K = unsafe {
                std::ptr::read_unaligned(row_bytes.as_ptr().add(off) as *const BlockQ6_K)
            };
            dequantize_q6_k(&block, &mut tmp);
            out[bi * 256..(bi + 1) * 256].copy_from_slice(&tmp);
        }
        out
    }
}
