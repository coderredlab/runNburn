#[cfg(feature = "mediatek")]
use super::dequant::dequantize_bytes_to_f32;
#[cfg(target_arch = "aarch64")]
use super::quantized_dispatch::{
    build_q80_f32_scales, pack_q80_row_pairs, q80_f32_scales_requested, q80_prepack_load_enabled,
    repack_q4k_artifacts,
};
#[cfg(feature = "mediatek")]
use crate::error::LlmError;
use crate::runtime::{QuantFormat, QuantizedWeightView};
use rnb_core::tensor::Tensor;
use rnb_loader::GGMLType;
#[cfg(feature = "mediatek")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(target_arch = "aarch64")]
use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
pub(super) struct ArchPackedWeightState {
    pub(super) repacked: Option<memmap2::Mmap>,
    pub(super) twin_repacked: Option<memmap2::Mmap>,
    pub(super) meta_repacked: Option<memmap2::Mmap>,
    pub(super) q80_pair_packed: OnceLock<Option<Vec<u8>>>,
    pub(super) q80_f32_scales: OnceLock<Option<Vec<f32>>>,
}

#[cfg(not(target_arch = "aarch64"))]
pub(super) struct ArchPackedWeightState;

impl ArchPackedWeightState {
    pub(super) fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                repacked: None,
                twin_repacked: None,
                meta_repacked: None,
                q80_pair_packed: OnceLock::new(),
                q80_f32_scales: OnceLock::new(),
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self
        }
    }
}

pub(super) fn backend_quant_format(ggml_type: GGMLType) -> Option<QuantFormat> {
    match ggml_type {
        GGMLType::F32 => Some(QuantFormat::F32),
        GGMLType::F16 => Some(QuantFormat::F16),
        GGMLType::BF16 => Some(QuantFormat::BF16),
        GGMLType::Q4_0 => Some(QuantFormat::Q40),
        GGMLType::Q4_1 => Some(QuantFormat::Q41),
        GGMLType::Q5_0 => Some(QuantFormat::Q50),
        GGMLType::Q5_1 => Some(QuantFormat::Q51),
        GGMLType::Q2_K => Some(QuantFormat::Q2K),
        GGMLType::Q3_K => Some(QuantFormat::Q3K),
        GGMLType::Q4_K => Some(QuantFormat::Q4K),
        GGMLType::Q5_K => Some(QuantFormat::Q5K),
        GGMLType::Q6_K => Some(QuantFormat::Q6K),
        GGMLType::Q8_0 => Some(QuantFormat::Q80),
        GGMLType::Q8_1 => Some(QuantFormat::Q81),
        GGMLType::IQ2_XXS => Some(QuantFormat::IQ2XXS),
        GGMLType::IQ2_S => Some(QuantFormat::IQ2S),
        GGMLType::IQ3_XXS => Some(QuantFormat::IQ3XXS),
        GGMLType::IQ4_XS => Some(QuantFormat::IQ4XS),
        GGMLType::Q8_K
        | GGMLType::IQ2_XS
        | GGMLType::IQ1_S
        | GGMLType::IQ4_NL
        | GGMLType::IQ3_S
        | GGMLType::IQ1_M
        | GGMLType::TQ1_0
        | GGMLType::TQ2_0
        | GGMLType::MXFP4
        | GGMLType::NVFP4
        | GGMLType::Q1_0
        | GGMLType::Q2_0
        | GGMLType::I32
        | GGMLType::I8
        | GGMLType::I16
        | GGMLType::I64
        | GGMLType::F64 => None,
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub(super) fn backend_ggml_type(quant: QuantFormat) -> GGMLType {
    match quant {
        QuantFormat::F32 => GGMLType::F32,
        QuantFormat::F16 => GGMLType::F16,
        QuantFormat::BF16 => GGMLType::BF16,
        QuantFormat::Q40 => GGMLType::Q4_0,
        QuantFormat::Q41 => GGMLType::Q4_1,
        QuantFormat::Q50 => GGMLType::Q5_0,
        QuantFormat::Q51 => GGMLType::Q5_1,
        QuantFormat::Q2K => GGMLType::Q2_K,
        QuantFormat::Q3K => GGMLType::Q3_K,
        QuantFormat::Q4K => GGMLType::Q4_K,
        QuantFormat::Q5K => GGMLType::Q5_K,
        QuantFormat::Q6K => GGMLType::Q6_K,
        QuantFormat::Q80 => GGMLType::Q8_0,
        QuantFormat::Q81 => GGMLType::Q8_1,
        QuantFormat::IQ2XXS => GGMLType::IQ2_XXS,
        QuantFormat::IQ2S => GGMLType::IQ2_S,
        QuantFormat::IQ3XXS => GGMLType::IQ3_XXS,
        QuantFormat::IQ4XS => GGMLType::IQ4_XS,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct QuantizedWeightDescriptor {
    rows: usize,
    cols: usize,
    ggml_type: GGMLType,
}

impl QuantizedWeightDescriptor {
    pub(super) const fn new(rows: usize, cols: usize, ggml_type: GGMLType) -> Self {
        Self {
            rows,
            cols,
            ggml_type,
        }
    }

    pub(super) const fn rows(self) -> usize {
        self.rows
    }

    pub(super) const fn cols(self) -> usize {
        self.cols
    }

    pub(super) const fn ggml_type(self) -> GGMLType {
        self.ggml_type
    }

    pub(super) fn quant_format(self) -> Option<QuantFormat> {
        backend_quant_format(self.ggml_type())
    }

    pub(super) fn supports_backend_raw_view(self) -> bool {
        self.quant_format().is_some()
    }

    pub(super) fn backend_view<'a>(self, raw: &'a [u8]) -> Option<QuantizedWeightView<'a>> {
        if !self.supports_backend_raw_view() {
            return None;
        }
        let quant = self.quant_format()?;
        Some(QuantizedWeightView::new(
            raw,
            self.rows(),
            self.cols(),
            quant,
        ))
    }
}

#[cfg(feature = "mediatek")]
static MEDIATEK_CACHE_GENERATION_ID: AtomicU64 = AtomicU64::new(1);

/// 양자화 상태의 weight. mmap에서 직접 참조하여 메모리 절약.
/// gemv 시 행 단위로 on-the-fly dequant -> dot product.
pub(super) struct QuantizedWeight {
    pub(super) descriptor: QuantizedWeightDescriptor,
    pub(super) data: Tensor, // 원본 양자화 Tensor (mmap 참조, U8/I8/F32/F16)
    pub(super) ggml_type: GGMLType,
    pub(super) rows: usize, // out_features
    pub(super) cols: usize, // in_features
    /// Q4_K -> Q4_0 변환 데이터 (simpler format, faster kernel)
    pub(super) q4_0_data: Option<Vec<u8>>,
    #[cfg(feature = "mediatek")]
    pub(super) mediatek_cache_generation_id: u64,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) arch: ArchPackedWeightState,
}

impl QuantizedWeight {
    pub(super) fn new(data: Tensor, ggml_type: GGMLType, rows: usize, cols: usize) -> Self {
        Self {
            descriptor: QuantizedWeightDescriptor::new(rows, cols, ggml_type),
            data,
            ggml_type,
            rows,
            cols,
            q4_0_data: None,
            #[cfg(feature = "mediatek")]
            mediatek_cache_generation_id: MEDIATEK_CACHE_GENERATION_ID
                .fetch_add(1, Ordering::Relaxed),
            arch: ArchPackedWeightState::new(),
        }
    }

    pub(super) fn descriptor(&self) -> QuantizedWeightDescriptor {
        self.descriptor
    }

    pub(super) fn backend_view(&self) -> Option<QuantizedWeightView<'_>> {
        let raw = self.data.as_bytes()?;
        self.descriptor().backend_view(raw)
    }

    #[cfg(feature = "mediatek")]
    pub(super) fn mediatek_gated_gelu_cache_weight_key(
        &self,
    ) -> Option<crate::engine::mediatek_runtime::GatedGeluFfnF32WeightKey> {
        let raw = self.data.as_bytes()?;
        Some(crate::engine::mediatek_runtime::GatedGeluFfnF32WeightKey {
            generation_id: self.mediatek_cache_generation_id,
            raw_ptr: raw.as_ptr() as usize,
            raw_len: raw.len(),
            rows: self.rows,
            cols: self.cols,
            ggml_type: self.ggml_type as u32,
        })
    }

    #[cfg(feature = "mediatek")]
    pub(super) fn materialize_f32_owned(
        &self,
        label: &'static str,
    ) -> crate::error::Result<Vec<f32>> {
        let bytes = self.data.as_bytes().ok_or_else(|| {
            LlmError::Forward(format!(
                "MediaTek FFN {label}: quantized weight bytes unavailable"
            ))
        })?;
        let expected = self.rows.checked_mul(self.cols).ok_or_else(|| {
            LlmError::Forward(format!(
                "MediaTek FFN {label}: f32 materialization length overflow"
            ))
        })?;
        let values = dequantize_bytes_to_f32(bytes, self.ggml_type);
        if values.len() != expected {
            return Err(LlmError::Forward(format!(
                "MediaTek FFN {label}: materialized len mismatch expected {expected}, got {}",
                values.len()
            )));
        }
        if let Some((idx, _)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(LlmError::Forward(format!(
                "MediaTek FFN {label}: non-finite materialized value at index {idx}"
            )));
        }
        Ok(values)
    }

    pub(super) fn new_with_q4k_repack(
        data: Tensor,
        ggml_type: GGMLType,
        rows: usize,
        cols: usize,
        bytes: Option<&[u8]>,
        should_repack: bool,
    ) -> Self {
        #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
        let mut weight = Self::new(data, ggml_type, rows, cols);
        #[cfg(target_arch = "aarch64")]
        if should_repack {
            if let Some(bytes) = bytes {
                let artifacts = repack_q4k_artifacts(bytes, rows, cols);
                weight.arch.repacked = Some(artifacts.repacked);
                weight.arch.twin_repacked = Some(artifacts.twin_repacked);
                weight.arch.meta_repacked = Some(artifacts.meta_repacked);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = (bytes, should_repack);
        weight
    }

    pub(super) fn new_q80_with_load_time_packs(
        data: Tensor,
        rows: usize,
        cols: usize,
        total_bytes: usize,
    ) -> Self {
        let weight = Self::new(data, GGMLType::Q8_0, rows, cols);
        #[cfg(target_arch = "aarch64")]
        {
            let bytes = weight
                .data
                .as_bytes()
                .expect("Q8_0 tensor must expose bytes");
            if q80_prepack_load_enabled() {
                let packed = pack_q80_row_pairs(bytes, rows, total_bytes / rows);
                eprintln!(
                    "[INFO] output: Q8_0 row-pair prepacked at load ({:.1}MB)",
                    packed.len() as f64 / 1e6
                );
                let _ = weight.arch.q80_pair_packed.set(Some(packed));
            }
            if q80_f32_scales_requested() {
                let scales = build_q80_f32_scales(bytes, rows, cols, total_bytes);
                eprintln!(
                    "[INFO] output: Q8_0 f32 scale sidecar at load ({:.1}MB)",
                    (scales.len() * std::mem::size_of::<f32>()) as f64 / 1e6
                );
                let _ = weight.arch.q80_f32_scales.set(Some(scales));
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = total_bytes;
        weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_builds_backend_view_from_supported_raw_bytes() {
        let descriptor = QuantizedWeightDescriptor::new(2, 3, GGMLType::Q6_K);
        let raw = [1u8, 2, 3, 4];

        let view = descriptor
            .backend_view(&raw)
            .expect("Q6_K has backend view");

        assert_eq!(view.raw(), &raw);
        assert_eq!(view.rows(), 2);
        assert_eq!(view.cols(), 3);
        assert_eq!(view.quant(), QuantFormat::Q6K);
    }

    #[test]
    fn descriptor_rejects_unsupported_backend_quant_format() {
        let descriptor = QuantizedWeightDescriptor::new(2, 3, GGMLType::I32);
        let raw = [1u8, 2, 3, 4];

        assert!(descriptor.backend_view(&raw).is_none());
    }

    #[test]
    fn descriptor_reports_raw_view_support_from_quant_mapping() {
        assert!(QuantizedWeightDescriptor::new(2, 3, GGMLType::Q4_K).supports_backend_raw_view());
        assert!(QuantizedWeightDescriptor::new(2, 3, GGMLType::Q4_0).supports_backend_raw_view());
        assert!(!QuantizedWeightDescriptor::new(2, 3, GGMLType::I32).supports_backend_raw_view());
    }

    #[test]
    fn quantized_weight_backend_view_uses_descriptor() {
        let weight = QuantizedWeight::new(
            Tensor::from_vec(vec![1u8, 2, 3, 4], &[4]),
            GGMLType::Q5_K,
            2,
            3,
        );
        let descriptor = weight.descriptor();

        assert_eq!(descriptor.rows(), 2);
        assert_eq!(descriptor.cols(), 3);
        assert_eq!(descriptor.ggml_type(), GGMLType::Q5_K);
        assert_eq!(
            weight.backend_view().map(|view| view.quant()),
            Some(QuantFormat::Q5K)
        );
    }
}
