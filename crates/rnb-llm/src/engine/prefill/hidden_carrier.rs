#[cfg(feature = "cuda")]
use crate::engine::backend_runtime;
use rnb_core::tensor::Tensor;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(dead_code)]
pub(in crate::engine) enum PrefillHidden {
    Host(Tensor),
    #[cfg(feature = "cuda")]
    Device(DevicePrefillHidden),
}

#[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) struct MetalQwenPrefillOwnershipReport {
    pub requested_layers: usize,
    pub attention_kv_writes: usize,
    pub gdn_state_writes: usize,
    pub hidden_uploads: usize,
    pub hidden_readbacks: usize,
    pub intermediate_hidden_transfers: usize,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::engine) struct MetalQwenPrefillOwnershipCounters {
    pub eligible_hits: usize,
    pub layer_hits: usize,
    pub attention_kv_writes: usize,
    pub gdn_state_writes: usize,
    pub hidden_uploads: usize,
    pub hidden_readbacks: usize,
    pub intermediate_hidden_transfers: usize,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
static METAL_QWEN_PREFILL_ELIGIBLE_HITS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(feature = "metal", not(feature = "cuda")))]
static METAL_QWEN_PREFILL_LAYER_HITS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(feature = "metal", not(feature = "cuda")))]
static METAL_QWEN_PREFILL_ATTENTION_KV_WRITES: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(feature = "metal", not(feature = "cuda")))]
static METAL_QWEN_PREFILL_GDN_STATE_WRITES: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(feature = "metal", not(feature = "cuda")))]
static METAL_QWEN_PREFILL_HIDDEN_UPLOADS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(feature = "metal", not(feature = "cuda")))]
static METAL_QWEN_PREFILL_HIDDEN_READBACKS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(feature = "metal", not(feature = "cuda")))]
static METAL_QWEN_PREFILL_INTERMEDIATE_HIDDEN_TRANSFERS: AtomicUsize = AtomicUsize::new(0);

#[cfg(any(all(feature = "metal", not(feature = "cuda")), test))]
pub(in crate::engine) fn validate_metal_qwen_prefill_ownership(
    requested_layers: usize,
    expected_attention_layers: usize,
    expected_gdn_layers: usize,
    attention_kv_writes: usize,
    gdn_state_writes: usize,
    hidden_uploads: usize,
    hidden_readbacks: usize,
    intermediate_hidden_transfers: usize,
) -> crate::error::Result<MetalQwenPrefillOwnershipReport> {
    if requested_layers == 0
        || expected_attention_layers + expected_gdn_layers != requested_layers
        || attention_kv_writes != expected_attention_layers
        || gdn_state_writes != expected_gdn_layers
        || hidden_uploads != 1
        || hidden_readbacks != 1
        || intermediate_hidden_transfers != 0
    {
        return Err(crate::error::LlmError::Forward(format!(
            "Metal Qwen prefill ownership mismatch: layers={requested_layers} expected_attention={expected_attention_layers} expected_gdn={expected_gdn_layers} attention_kv={attention_kv_writes} gdn_states={gdn_state_writes} hidden_uploads={hidden_uploads} hidden_readbacks={hidden_readbacks} intermediate_hidden_transfers={intermediate_hidden_transfers}"
        )));
    }
    Ok(MetalQwenPrefillOwnershipReport {
        requested_layers,
        attention_kv_writes,
        gdn_state_writes,
        hidden_uploads,
        hidden_readbacks,
        intermediate_hidden_transfers,
    })
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(in crate::engine) fn record_metal_qwen_prefill_ownership(
    report: MetalQwenPrefillOwnershipReport,
) {
    let eligible_hits = METAL_QWEN_PREFILL_ELIGIBLE_HITS.fetch_add(1, Ordering::Relaxed) + 1;
    METAL_QWEN_PREFILL_LAYER_HITS.fetch_add(report.requested_layers, Ordering::Relaxed);
    METAL_QWEN_PREFILL_ATTENTION_KV_WRITES.fetch_add(report.attention_kv_writes, Ordering::Relaxed);
    METAL_QWEN_PREFILL_GDN_STATE_WRITES.fetch_add(report.gdn_state_writes, Ordering::Relaxed);
    METAL_QWEN_PREFILL_HIDDEN_UPLOADS.fetch_add(report.hidden_uploads, Ordering::Relaxed);
    METAL_QWEN_PREFILL_HIDDEN_READBACKS.fetch_add(report.hidden_readbacks, Ordering::Relaxed);
    METAL_QWEN_PREFILL_INTERMEDIATE_HIDDEN_TRANSFERS
        .fetch_add(report.intermediate_hidden_transfers, Ordering::Relaxed);
    let trace_enabled = std::env::var("RNB_METAL_QWEN_PREFILL_CHAIN_TRACE")
        .ok()
        .is_some_and(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        });
    if trace_enabled {
        eprintln!(
            "[metal:qwen-prefill-chain] eligible_hit=1 eligible_hits_total={eligible_hits} layers={} attention_kv={} gdn_states={} hidden_uploads={} hidden_readbacks={} intermediate_hidden_transfers={}",
            report.requested_layers,
            report.attention_kv_writes,
            report.gdn_state_writes,
            report.hidden_uploads,
            report.hidden_readbacks,
            report.intermediate_hidden_transfers,
        );
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(dead_code)]
pub(in crate::engine) fn metal_qwen_prefill_ownership_counters() -> MetalQwenPrefillOwnershipCounters
{
    MetalQwenPrefillOwnershipCounters {
        eligible_hits: METAL_QWEN_PREFILL_ELIGIBLE_HITS.load(Ordering::Relaxed),
        layer_hits: METAL_QWEN_PREFILL_LAYER_HITS.load(Ordering::Relaxed),
        attention_kv_writes: METAL_QWEN_PREFILL_ATTENTION_KV_WRITES.load(Ordering::Relaxed),
        gdn_state_writes: METAL_QWEN_PREFILL_GDN_STATE_WRITES.load(Ordering::Relaxed),
        hidden_uploads: METAL_QWEN_PREFILL_HIDDEN_UPLOADS.load(Ordering::Relaxed),
        hidden_readbacks: METAL_QWEN_PREFILL_HIDDEN_READBACKS.load(Ordering::Relaxed),
        intermediate_hidden_transfers: METAL_QWEN_PREFILL_INTERMEDIATE_HIDDEN_TRANSFERS
            .load(Ordering::Relaxed),
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(dead_code)]
pub(in crate::engine) fn reset_metal_qwen_prefill_ownership_counters() {
    METAL_QWEN_PREFILL_ELIGIBLE_HITS.store(0, Ordering::Relaxed);
    METAL_QWEN_PREFILL_LAYER_HITS.store(0, Ordering::Relaxed);
    METAL_QWEN_PREFILL_ATTENTION_KV_WRITES.store(0, Ordering::Relaxed);
    METAL_QWEN_PREFILL_GDN_STATE_WRITES.store(0, Ordering::Relaxed);
    METAL_QWEN_PREFILL_HIDDEN_UPLOADS.store(0, Ordering::Relaxed);
    METAL_QWEN_PREFILL_HIDDEN_READBACKS.store(0, Ordering::Relaxed);
    METAL_QWEN_PREFILL_INTERMEDIATE_HIDDEN_TRANSFERS.store(0, Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) struct DevicePrefillHidden {
    pub output: backend_runtime::NemotronDeviceLayerOutput,
    pub producer_layer_idx: usize,
}

#[cfg(feature = "cuda")]
impl DevicePrefillHidden {
    #[allow(dead_code)]
    pub(in crate::engine) fn hidden_bytes(&self) -> crate::error::Result<usize> {
        self.output.output_desc.byte_len().ok_or_else(|| {
            crate::error::LlmError::Forward("CUDA device hidden byte length overflow".to_string())
        })
    }
}

impl PrefillHidden {
    #[allow(dead_code)]
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub(in crate::engine) fn into_host_for_layer(
        self,
        consumer_layer_idx: Option<usize>,
        reason: &'static str,
    ) -> crate::error::Result<Tensor> {
        match self {
            Self::Host(hidden) => Ok(hidden),
            #[cfg(feature = "cuda")]
            Self::Device(device) => materialize_device_hidden(device, consumer_layer_idx, reason),
        }
    }

    #[allow(dead_code)]
    pub(in crate::engine) fn as_host(&self) -> Option<&Tensor> {
        match self {
            Self::Host(hidden) => Some(hidden),
            #[cfg(feature = "cuda")]
            Self::Device(_) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(in crate::engine) fn take_device(self) -> Option<DevicePrefillHidden> {
        match self {
            Self::Device(device) => Some(device),
            Self::Host(_) => None,
        }
    }
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn materialize_device_hidden_row(
    device: DevicePrefillHidden,
    row: usize,
    consumer_layer_idx: Option<usize>,
    reason: &'static str,
) -> crate::error::Result<Tensor> {
    let output_desc = device.output.output_desc;
    if row >= output_desc.rows() {
        return Err(crate::error::LlmError::Forward(format!(
            "CUDA device hidden row out of range: row={}, rows={}",
            row,
            output_desc.rows()
        )));
    }
    let d2h_bytes = output_desc
        .cols()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            crate::error::LlmError::Forward(
                "CUDA device hidden row byte length overflow".to_string(),
            )
        })?;
    emit_device_hidden_materialize_trace(
        device.producer_layer_idx,
        consumer_layer_idx,
        d2h_bytes,
        reason,
    );
    let cols = output_desc.cols();
    let output = backend_runtime::download_nemotron_device_layer_output_row(device.output, row)?;
    Ok(Tensor::from_vec(output, &[1, cols]))
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(in crate::engine) fn materialize_device_hidden(
    device: DevicePrefillHidden,
    consumer_layer_idx: Option<usize>,
    reason: &'static str,
) -> crate::error::Result<Tensor> {
    let bytes = device.hidden_bytes()?;
    emit_device_hidden_materialize_trace(
        device.producer_layer_idx,
        consumer_layer_idx,
        bytes,
        reason,
    );
    let rows = device.output.output_desc.rows();
    let cols = device.output.output_desc.cols();
    let output = backend_runtime::download_nemotron_device_layer_output(device.output)?;
    Ok(Tensor::from_vec(output, &[rows, cols]))
}

#[cfg(feature = "cuda")]
fn emit_device_hidden_materialize_trace(
    producer_layer_idx: usize,
    consumer_layer_idx: Option<usize>,
    d2h_bytes: usize,
    reason: &'static str,
) {
    if !crate::engine::models::nemotron::mamba::device_prefill_trace_enabled() {
        return;
    }
    let consumer = consumer_layer_idx
        .map(|idx| idx.to_string())
        .unwrap_or_else(|| "none".to_string());
    eprintln!(
        "[cuda:device-prefill-chain] op=device_hidden_materialize layers={},{} d2h_bytes={} reason={}",
        producer_layer_idx, consumer, d2h_bytes, reason
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_carrier_materializes_without_transfer_bytes() {
        use crate::engine::cpu_runtime::kernels;

        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let hidden = PrefillHidden::Host(tensor);
        let materialized = hidden.into_host_for_layer(None, "range_end").unwrap();

        assert_eq!(materialized.shape(), &[2, 2]);
        assert_eq!(
            kernels::tensor_as_f32_slice(&materialized),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }
}
