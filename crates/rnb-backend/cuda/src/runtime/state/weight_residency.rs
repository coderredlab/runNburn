use super::super::*;

#[cfg(test)]
pub(in crate::runtime) const Q4K_RAW_BYTES_PER_BLOCK: usize = 144;
pub(in crate::runtime) const Q4K_PACKED_Q8DOT_BYTES_PER_BLOCK: usize = 148;
#[cfg(test)]
pub(in crate::runtime) const Q6K_RAW_BYTES_PER_BLOCK: usize = 210;
pub(in crate::runtime) const Q6K_PACKED_Q8DOT_BYTES_PER_BLOCK: usize = 274;

fn log_weight_residency(kind: &str, source: &str, dtype: &str, bytes: usize) {
    if std::env::var("RNB_CUDA_WEIGHT_RESIDENCY_LOG")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[cuda-weight-residency] residency={kind} source_quant={source} dtype={dtype} bytes={bytes}"
        );
    }
}

impl CudaState {
    pub(in crate::runtime) fn record_q4_expanded_f16(&mut self, bytes: usize) {
        self.weight_residency_counters.record_q4_expanded_f16(bytes);
        log_weight_residency("ExpandedDiag", "Q4_K", "F16", bytes);
    }

    pub(in crate::runtime) fn record_q4_expanded_f32(&mut self, bytes: usize) {
        self.weight_residency_counters.record_q4_expanded_f32(bytes);
        log_weight_residency("ExpandedDiag", "Q4_K", "F32", bytes);
    }

    pub(in crate::runtime) fn record_q6_expanded_f16(&mut self, bytes: usize) {
        self.weight_residency_counters.record_q6_expanded_f16(bytes);
        log_weight_residency("ExpandedDiag", "Q6_K", "F16", bytes);
    }

    pub(in crate::runtime) fn record_q6_expanded_f32(&mut self, bytes: usize) {
        self.weight_residency_counters.record_q6_expanded_f32(bytes);
        log_weight_residency("ExpandedDiag", "Q6_K", "F32", bytes);
    }

    pub(in crate::runtime) fn record_native_f32_residency(&mut self, bytes: usize) {
        self.weight_residency_counters.record_native_f32(bytes);
        log_weight_residency("NativeF32", "NativeF32", "F32", bytes);
    }

    pub(in crate::runtime) fn record_packed_q8dot_residency(
        &mut self,
        source_quant: &str,
        bytes: usize,
    ) {
        self.weight_residency_counters.record_packed_q8dot(bytes);
        log_weight_residency("PackedQ8Dot", source_quant, "packed", bytes);
    }

    pub(in crate::runtime) fn record_raw_quant_residency(
        &mut self,
        source_quant: &str,
        bytes: usize,
    ) {
        match source_quant {
            "Q4_K" => self.weight_residency_counters.record_q4_raw_quant(bytes),
            "Q6_K" => self.weight_residency_counters.record_q6_raw_quant(bytes),
            other => {
                log_weight_residency("RawQuant", other, "raw", bytes);
                return;
            }
        }
        log_weight_residency("RawQuant", source_quant, "raw", bytes);
    }

    pub(in crate::runtime) fn record_transient_quant_upload(
        &mut self,
        source_quant: &str,
        bytes: usize,
    ) {
        match source_quant {
            "Q4_K" => self
                .weight_residency_counters
                .record_q4_transient_quant_upload(bytes),
            "Q6_K" => self
                .weight_residency_counters
                .record_q6_transient_quant_upload(bytes),
            _ => {}
        }
        log_weight_residency("TransientUpload", source_quant, "raw", bytes);
    }

    pub(in crate::runtime) fn weight_residency_counters(&self) -> CudaWeightResidencyCounters {
        self.weight_residency_counters
    }
}

pub(in crate::runtime) fn validate_q4k_packed_payload_bytes_per_block(
    bytes: usize,
) -> Result<(), String> {
    if bytes == Q4K_PACKED_Q8DOT_BYTES_PER_BLOCK {
        Ok(())
    } else {
        Err(format!(
            "Q4_K packed payload must be {Q4K_PACKED_Q8DOT_BYTES_PER_BLOCK} bytes/block, got {bytes}"
        ))
    }
}

pub(in crate::runtime) fn validate_q6k_packed_payload_bytes_per_block(
    bytes: usize,
) -> Result<(), String> {
    if bytes == Q6K_PACKED_Q8DOT_BYTES_PER_BLOCK {
        Ok(())
    } else {
        Err(format!(
            "Q6_K packed payload must be {Q6K_PACKED_Q8DOT_BYTES_PER_BLOCK} bytes/block, got {bytes}"
        ))
    }
}

#[cfg(test)]
pub fn q4k_raw_bytes_per_block_for_test() -> usize {
    Q4K_RAW_BYTES_PER_BLOCK
}

#[cfg(test)]
pub fn q4k_packed_q8dot_bytes_per_block_for_test() -> usize {
    Q4K_PACKED_Q8DOT_BYTES_PER_BLOCK
}

#[cfg(test)]
pub fn q6k_raw_bytes_per_block_for_test() -> usize {
    Q6K_RAW_BYTES_PER_BLOCK
}

#[cfg(test)]
pub fn q6k_packed_q8dot_bytes_per_block_for_test() -> usize {
    Q6K_PACKED_Q8DOT_BYTES_PER_BLOCK
}

#[cfg(test)]
pub fn validate_q4k_packed_payload_bytes_for_test(bytes: usize) -> Result<(), String> {
    validate_q4k_packed_payload_bytes_per_block(bytes)
}

#[cfg(test)]
pub fn validate_q6k_packed_payload_bytes_for_test(bytes: usize) -> Result<(), String> {
    validate_q6k_packed_payload_bytes_per_block(bytes)
}
