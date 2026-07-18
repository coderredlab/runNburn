use rnb_backend_vulkan::staging::StagingPolicy;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn staging_scales_linearly_with_seq_len() {
    let _guard = ENV_LOCK.lock().unwrap();
    let p = StagingPolicy::default();
    let small = p.bytes_for(64, 1024);
    let large = p.bytes_for(4096, 1024);
    assert!(
        large >= small * 60,
        "staging must scale ~linearly: small={small} large={large}"
    );
}

#[test]
fn staging_includes_per_layer_intermediate_reserve() {
    let _guard = ENV_LOCK.lock().unwrap();
    let p = StagingPolicy::default();
    let raw = 393usize * 1024 * 2; // f16 input
    let bytes = p.bytes_for(393, 1024);
    assert!(bytes >= raw, "must reserve at least raw input size");
    assert!(bytes <= raw * 4, "reserve cap (4×) to avoid runaway");
}

#[test]
fn staging_respects_env_override() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("RNB_GPU_STAGING_RESERVE", "8192");
    let p = StagingPolicy::default();
    assert_eq!(p.reserve_bytes, 8192);
    std::env::remove_var("RNB_GPU_STAGING_RESERVE");
}
