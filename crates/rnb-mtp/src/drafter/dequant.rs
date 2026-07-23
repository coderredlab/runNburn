//! Drafter helper — quantized `TensorView` → row-major `Vec<f32>` 변환.
//!
//! cross_attention 과 vq_head 가 둘 다 매 forward 마다 dequant 해서 GEMV 에
//! 넣으므로 같은 헬퍼를 공유한다. block 단위로 `BlockQ4_K` / `BlockQ6_K` 를
//! `read_unaligned` 한 뒤 `rnb_cpu::quantize::dequant::*` 의 scalar 경로로
//! 풀어낸다 (mt82 forward.rs 의 패턴과 동일).
//!
//! Allocation: 호출마다 `Vec<f32>` 를 새로 할당한다. mt83 Stage C 까지는
//! 정확도 우선이고 dequant 성능 최적화는 Stage D 이후 calibration 결과에
//! 따라 정한다.

use super::types::TensorView;
use rnb_cpu::quantize::blocks::{BlockQ4_K, BlockQ6_K};
use rnb_cpu::quantize::dequant::{dequantize_q4_k, dequantize_q6_k};
use rnb_loader::gguf::types::GGMLType;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, OnceLock, RwLock};

// mc78: drafter forward 마다 ~22 dequant 호출 (9 weight × multi-step). 매번
// scalar dequant + Vec 할당 cost 큼 (PC CUDA E2B 의 drafter wall 7.5s/167rd ≈
// 45ms/rd, ~11ms per single drafter forward). weight 는 immutable 이라 (gguf
// mmap) ptr+len 키로 한 번 dequant 후 영구 cache 안전.
type DequantKey = (usize, usize); // (ptr, len)
fn dequant_cache() -> &'static RwLock<HashMap<DequantKey, Arc<Vec<f32>>>> {
    static CACHE: OnceLock<RwLock<HashMap<DequantKey, Arc<Vec<f32>>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Dequantize a `TensorView` to a row-major f32 `Vec`.
///
/// Supported dtypes: F32 (직접 복사), Q4_K, Q6_K. 그 외 dtype 은 drafter 의
/// 51 tensor 목록에 없으므로 panic.
///
/// mc78: 같은 TensorView (ptr+len 동일) 의 결과는 static cache 에서 hit.
pub(crate) fn dequant_to_f32(view: &TensorView) -> Vec<f32> {
    let bytes = view.as_bytes();
    let key: DequantKey = (bytes.as_ptr() as usize, bytes.len());
    if let Some(cached) = dequant_cache().read().unwrap().get(&key).cloned() {
        return (*cached).clone();
    }
    let result = dequant_to_f32_uncached(view);
    let arc = Arc::new(result.clone());
    dequant_cache().write().unwrap().insert(key, arc);
    result
}

fn dequant_to_f32_uncached(view: &TensorView) -> Vec<f32> {
    let elem_count: usize = view.shape.iter().product();
    let bytes = view.as_bytes();
    let mut out = vec![0.0f32; elem_count];

    match view.ggml_type {
        GGMLType::F32 => {
            assert_eq!(
                bytes.len(),
                elem_count * 4,
                "F32 byte length mismatch for {:?}",
                view.shape
            );
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GGMLType::Q4_K => {
            assert!(
                elem_count % 256 == 0,
                "Q4_K elem_count {elem_count} not multiple of 256"
            );
            let n_blocks = elem_count / 256;
            assert_eq!(
                bytes.len(),
                n_blocks * size_of::<BlockQ4_K>(),
                "Q4_K byte length mismatch"
            );
            let mut tmp = [0.0f32; 256];
            for bi in 0..n_blocks {
                let offset = bi * size_of::<BlockQ4_K>();
                let block: BlockQ4_K = unsafe {
                    std::ptr::read_unaligned(bytes.as_ptr().add(offset) as *const BlockQ4_K)
                };
                dequantize_q4_k(&block, &mut tmp);
                out[bi * 256..(bi + 1) * 256].copy_from_slice(&tmp);
            }
        }
        GGMLType::Q6_K => {
            assert!(
                elem_count % 256 == 0,
                "Q6_K elem_count {elem_count} not multiple of 256"
            );
            let n_blocks = elem_count / 256;
            assert_eq!(
                bytes.len(),
                n_blocks * size_of::<BlockQ6_K>(),
                "Q6_K byte length mismatch"
            );
            let mut tmp = [0.0f32; 256];
            for bi in 0..n_blocks {
                let offset = bi * size_of::<BlockQ6_K>();
                let block: BlockQ6_K = unsafe {
                    std::ptr::read_unaligned(bytes.as_ptr().add(offset) as *const BlockQ6_K)
                };
                dequantize_q6_k(&block, &mut tmp);
                out[bi * 256..(bi + 1) * 256].copy_from_slice(&tmp);
            }
        }
        other => panic!("dequant_to_f32: unsupported dtype {other:?}"),
    }

    out
}
