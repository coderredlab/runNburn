use rnb_backend_vulkan::spirv::emit_embed_lookup_q6k;
use rnb_backend_vulkan::{GpuWeightMode, VulkanLayerGemv};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn vulkan_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("vulkan test lock poisoned")
}

#[test]
fn emit_embed_lookup_q6k_word_format() {
    let spirv = emit_embed_lookup_q6k(64);
    eprintln!(
        "emit_embed_lookup_q6k: {} words ({} bytes)",
        spirv.len(),
        spirv.len() * 4
    );
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "shader too short: {}", spirv.len());
}

#[test]
fn embed_lookup_matches_cpu_dequant() {
    let _guard = vulkan_test_lock();
    let mut gemv = match VulkanLayerGemv::new(1024, 4096, 64, GpuWeightMode::Soa) {
        Ok(g) => g,
        Err(_) => return,
    };

    let vocab = 32usize;
    let hidden = 256usize;

    let q6k_table = build_dummy_q6k_table(vocab, hidden);
    let token_ids: Vec<u32> = vec![0, 5, 17, 31];

    let mut gpu_out = vec![0.0f32; token_ids.len() * hidden];
    gemv.embed_lookup(
        &token_ids,
        hidden as u32,
        vocab as u32,
        &q6k_table,
        &mut gpu_out,
    )
    .expect("embed_lookup");

    let cpu_out = cpu_q6k_dequant_rows(&q6k_table, &token_ids, hidden);

    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-3, "max_diff = {max_diff}");
}

#[test]
fn embed_lookup_token_zero_returns_first_row() {
    let _guard = vulkan_test_lock();
    let mut gemv = match VulkanLayerGemv::new(1024, 4096, 64, GpuWeightMode::Soa) {
        Ok(g) => g,
        Err(_) => return,
    };

    let vocab = 4usize;
    let hidden = 256usize;
    let q6k_table = build_dummy_q6k_table(vocab, hidden);

    let token_ids = vec![0u32];
    let mut gpu_out = vec![0.0f32; hidden];
    gemv.embed_lookup(
        &token_ids,
        hidden as u32,
        vocab as u32,
        &q6k_table,
        &mut gpu_out,
    )
    .expect("embed_lookup");

    let block: rnb_cpu::quantize::blocks::BlockQ6_K = read_block(&q6k_table, 0);
    let mut cpu_out = [0.0f32; 256];
    rnb_cpu::quantize::dequant::dequantize_q6_k(&block, &mut cpu_out);

    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-3, "max_diff = {max_diff}");
}

#[test]
fn embed_lookup_oob_token_id_writes_zeros() {
    let _guard = vulkan_test_lock();
    let mut gemv = match VulkanLayerGemv::new(1024, 4096, 64, GpuWeightMode::Soa) {
        Ok(g) => g,
        Err(_) => return,
    };

    let vocab = 4usize;
    let hidden = 256usize;
    let q6k_table = build_dummy_q6k_table(vocab, hidden);

    // token_id 999 is out-of-bounds for vocab=4; guard must write 0.0
    let token_ids = vec![999u32];
    let mut gpu_out = vec![f32::NAN; hidden];
    gemv.embed_lookup(
        &token_ids,
        hidden as u32,
        vocab as u32,
        &q6k_table,
        &mut gpu_out,
    )
    .expect("embed_lookup");

    for (i, &v) in gpu_out.iter().enumerate() {
        assert_eq!(v, 0.0f32, "expected 0.0 at index {i}, got {v}");
    }
}

fn build_dummy_q6k_table(vocab: usize, hidden: usize) -> Vec<u8> {
    use rnb_cpu::quantize::blocks::BlockQ6_K;
    use std::mem::size_of;
    assert_eq!(hidden % 256, 0);
    let blocks_per_row = hidden / 256;
    let mut bytes = vec![0u8; vocab * blocks_per_row * size_of::<BlockQ6_K>()];
    for v in 0..vocab {
        for b in 0..blocks_per_row {
            let block = BlockQ6_K {
                ql: std::array::from_fn(|i| ((v * 31 + b * 7 + i) % 256) as u8),
                qh: std::array::from_fn(|i| ((v * 17 + b * 3 + i) % 256) as u8),
                scales: std::array::from_fn(|i| {
                    ((v as i32 * 5 + b as i32 + i as i32) % 64 - 32) as i8
                }),
                d: half::f16::from_f32(0.1 + (v as f32 + b as f32) * 0.01),
            };
            let off = (v * blocks_per_row + b) * size_of::<BlockQ6_K>();
            let src = unsafe {
                std::slice::from_raw_parts(
                    &block as *const BlockQ6_K as *const u8,
                    size_of::<BlockQ6_K>(),
                )
            };
            bytes[off..off + size_of::<BlockQ6_K>()].copy_from_slice(src);
        }
    }
    bytes
}

fn cpu_q6k_dequant_rows(table: &[u8], ids: &[u32], hidden: usize) -> Vec<f32> {
    use rnb_cpu::quantize::blocks::BlockQ6_K;
    use rnb_cpu::quantize::dequant::dequantize_q6_k;
    use std::mem::size_of;
    assert_eq!(hidden % 256, 0);
    let blocks_per_row = hidden / 256;
    let row_bytes = blocks_per_row * size_of::<BlockQ6_K>();
    let mut out = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        let row_off = (id as usize) * row_bytes;
        for b in 0..blocks_per_row {
            let block_off = row_off + b * size_of::<BlockQ6_K>();
            let block: BlockQ6_K = read_block(table, block_off / size_of::<BlockQ6_K>());
            let mut buf = [0.0f32; 256];
            dequantize_q6_k(&block, &mut buf);
            out.extend_from_slice(&buf);
        }
    }
    out
}

fn read_block(table: &[u8], block_idx: usize) -> rnb_cpu::quantize::blocks::BlockQ6_K {
    use rnb_cpu::quantize::blocks::BlockQ6_K;
    use std::mem::size_of;
    let off = block_idx * size_of::<BlockQ6_K>();
    let mut block = std::mem::MaybeUninit::<BlockQ6_K>::zeroed();
    unsafe {
        std::ptr::copy_nonoverlapping(
            table.as_ptr().add(off),
            block.as_mut_ptr() as *mut u8,
            size_of::<BlockQ6_K>(),
        );
        block.assume_init()
    }
}
