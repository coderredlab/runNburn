use rnb_backend_vulkan::spirv::emit_kv_append;
use rnb_backend_vulkan::{GpuWeightMode, VulkanLayerGemv};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn vulkan_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("vulkan test lock poisoned")
}

#[test]
fn emit_kv_append_word_format() {
    let spirv = emit_kv_append(64);
    eprintln!(
        "emit_kv_append: {} words ({} bytes)",
        spirv.len(),
        spirv.len() * 4
    );
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "shader too short: {}", spirv.len());
}

#[test]
fn kv_append_layer0_cursor0_writes_correct_offset() {
    let _guard = vulkan_test_lock();
    let mut gemv = match VulkanLayerGemv::new(1024, 4096, 64, GpuWeightMode::Soa) {
        Ok(g) => g,
        Err(_) => return,
    };

    let kv_heads = 2u32;
    let head_dim = 4u32;
    let stripe = (kv_heads * head_dim) as usize;
    let num_layers = 24u32;
    let max_ctx = 4096u32;
    let total = (num_layers as usize) * (max_ctx as usize) * 2 * stripe;

    let k_in: Vec<f32> = (0..stripe).map(|i| (i + 1) as f32).collect();
    let v_in: Vec<f32> = (0..stripe).map(|i| (i + 1) as f32 * 10.0).collect();
    let mut kv_buffer = vec![0.0f32; total];

    gemv.kv_append(
        0,
        0,
        &k_in,
        &v_in,
        &mut kv_buffer,
        num_layers,
        max_ctx,
        kv_heads,
        head_dim,
    )
    .expect("kv_append");

    assert_eq!(&kv_buffer[..stripe], &k_in[..]);
    assert_eq!(&kv_buffer[stripe..2 * stripe], &v_in[..]);
    assert_eq!(kv_buffer[2 * stripe], 0.0, "first untouched position");
}

#[test]
fn kv_append_layer1_cursor3_writes_correct_offset() {
    let _guard = vulkan_test_lock();
    let mut gemv = match VulkanLayerGemv::new(1024, 4096, 64, GpuWeightMode::Soa) {
        Ok(g) => g,
        Err(_) => return,
    };

    let kv_heads = 2u32;
    let head_dim = 4u32;
    let stripe = (kv_heads * head_dim) as usize;
    let num_layers = 24u32;
    let max_ctx = 4096u32;
    let total = (num_layers as usize) * (max_ctx as usize) * 2 * stripe;

    let k_in: Vec<f32> = (0..stripe).map(|i| (i + 1) as f32).collect();
    let v_in: Vec<f32> = (0..stripe).map(|i| (i + 1) as f32 * 10.0).collect();
    let mut kv_buffer = vec![0.0f32; total];

    let layer = 1u32;
    let cursor = 3u32;
    gemv.kv_append(
        layer,
        cursor,
        &k_in,
        &v_in,
        &mut kv_buffer,
        num_layers,
        max_ctx,
        kv_heads,
        head_dim,
    )
    .expect("kv_append");

    let per_layer = (max_ctx as usize) * 2 * stripe;
    let per_token = 2 * stripe;
    let k_off = (layer as usize) * per_layer + (cursor as usize) * per_token;
    assert_eq!(&kv_buffer[k_off..k_off + stripe], &k_in[..]);
    assert_eq!(&kv_buffer[k_off + stripe..k_off + 2 * stripe], &v_in[..]);
}
