use rnb_backend_vulkan::kv_resident::KvResidentLayout;

#[test]
fn qwen35_08b_max_ctx_4096_is_192mb() {
    let layout = KvResidentLayout::compute(24, 4096, 2, 256);
    assert_eq!(layout.total_bytes(), 24 * 4096 * 2 * 2 * 256 * 2);
    assert_eq!(layout.total_bytes() / (1024 * 1024), 192);
}

#[test]
fn offset_layer_zero_cursor_zero() {
    let layout = KvResidentLayout::compute(24, 4096, 2, 256);
    assert_eq!(layout.k_offset_bytes(0, 0), 0);
}

#[test]
fn offset_per_layer_increments_correctly() {
    let layout = KvResidentLayout::compute(24, 4096, 2, 256);
    let layer1_start = layout.k_offset_bytes(1, 0);
    let layer0_full = 4096 * 2 * 2 * 256 * 2; // ctx * (k+v) * heads * dim * f16
    assert_eq!(layer1_start, layer0_full);
}

#[test]
fn max_ctx_resolves_from_env_or_default() {
    std::env::set_var("RNB_MAX_CTX", "512");
    let m = KvResidentLayout::resolve_max_ctx(4096);
    assert_eq!(m, 512);
    std::env::remove_var("RNB_MAX_CTX");
    let m2 = KvResidentLayout::resolve_max_ctx(4096);
    assert_eq!(m2, 4096);
}
