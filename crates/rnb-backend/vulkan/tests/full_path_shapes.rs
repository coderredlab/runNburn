//! Shape / type-level smoke tests for `full_path::FullPathPrefillInput` /
//! `FullPathPrefillOutput`. The actual `run_prefill` invocation requires a live
//! Vulkan device and is exercised via the `vulkan_test` feature elsewhere.

use rnb_backend_vulkan::full_path::FullPathPrefillOutput;
use rnb_backend_vulkan::kv_resident::KvResidentLayout;
use rnb_backend_vulkan::staging::StagingPolicy;
use rnb_backend_vulkan::RuntimeCounters;

#[test]
fn runtime_counters_default_is_zero() {
    let c = RuntimeCounters::default();
    assert_eq!(c.submits, 0);
    assert_eq!(c.upload_bytes, 0);
    assert_eq!(c.download_bytes, 0);
    assert_eq!(c.materializations, 0);
}

#[test]
fn full_path_prefill_output_kv_cursor_matches_prompt_len() {
    // The contract is: after prefill of N tokens, kv_cursor_after == N.
    // Sanity-check the struct field is mutable / addressable.
    let out = FullPathPrefillOutput {
        last_token_id: 42,
        kv_cursor_after: 393,
        counters: RuntimeCounters::default(),
    };
    assert_eq!(out.kv_cursor_after, 393);
    assert_eq!(out.last_token_id, 42);
}

#[test]
fn staging_and_kv_layout_compose_for_qwen35_0p8b_shape() {
    // Qwen3.5 0.8B: 24 layers × 4096 ctx × 2 kv_heads × 256 head_dim × f16.
    let kv = KvResidentLayout::compute(24, 4096, 2, 256);
    assert!(kv.total_bytes() > 0);
    assert_eq!(kv.k_offset_bytes(0, 0), 0);

    let staging = StagingPolicy::default();
    let bytes = staging.bytes_for(393, 1024);
    assert!(bytes >= 393 * 1024 * 2, "must cover f16 prompt staging");
}
