use rnb_backend_vulkan::spirv::{emit_logit_argmax_q6k, emit_logit_argmax_q8_0};
use rnb_backend_vulkan::{GpuWeightMode, QuantType, VulkanLayerGemv};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn vulkan_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("vulkan test lock poisoned")
}

#[test]
fn emit_logit_argmax_q6k_word_format() {
    let spirv = emit_logit_argmax_q6k(256);
    eprintln!(
        "emit_logit_argmax_q6k: {} words ({} bytes)",
        spirv.len(),
        spirv.len() * 4
    );
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "shader too short: {}", spirv.len());
}

#[test]
fn emit_logit_argmax_q8_0_word_format() {
    let spirv = emit_logit_argmax_q8_0(256);
    eprintln!(
        "emit_logit_argmax_q8_0: {} words ({} bytes)",
        spirv.len(),
        spirv.len() * 4
    );
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic");
    assert_eq!(spirv[1], 0x00010300, "SPIR-V version 1.3");
    assert!(spirv.len() > 100, "shader too short: {}", spirv.len());
}

#[test]
fn logit_argmax_matches_cpu_small() {
    let _guard = vulkan_test_lock();
    let mut gemv = match VulkanLayerGemv::new(1024, 4096, 64, GpuWeightMode::Soa) {
        Ok(g) => g,
        Err(_) => return,
    };

    let vocab = 16usize;
    let hidden = 256usize;

    let q6k_table = build_dummy_q6k_table(vocab, hidden);
    let hidden_vec: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.001 - 0.128).sin())
        .collect();

    let cpu_argmax = cpu_q6k_matmul_argmax(&q6k_table, &hidden_vec, vocab, hidden);
    let gpu_argmax = gemv
        .logit_argmax(
            &hidden_vec,
            &q6k_table,
            QuantType::Q6K,
            vocab as u32,
            hidden as u32,
        )
        .expect("logit_argmax");
    assert_eq!(gpu_argmax, cpu_argmax, "gpu={gpu_argmax} cpu={cpu_argmax}");
}

#[test]
fn logit_argmax_larger_vocab() {
    let _guard = vulkan_test_lock();
    let mut gemv = match VulkanLayerGemv::new(1024, 4096, 64, GpuWeightMode::Soa) {
        Ok(g) => g,
        Err(_) => return,
    };

    let vocab = 1024usize;
    let hidden = 256usize;
    let q6k_table = build_dummy_q6k_table(vocab, hidden);
    let hidden_vec: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.0123).cos()).collect();

    let cpu_argmax = cpu_q6k_matmul_argmax(&q6k_table, &hidden_vec, vocab, hidden);
    let gpu_argmax = gemv
        .logit_argmax(
            &hidden_vec,
            &q6k_table,
            QuantType::Q6K,
            vocab as u32,
            hidden as u32,
        )
        .expect("logit_argmax");
    assert_eq!(gpu_argmax, cpu_argmax, "gpu={gpu_argmax} cpu={cpu_argmax}");
}

#[test]
fn logit_argmax_q8_0_matches_cpu_small() {
    let _guard = vulkan_test_lock();
    let mut gemv = match VulkanLayerGemv::new(1024, 4096, 64, GpuWeightMode::Soa) {
        Ok(g) => g,
        Err(_) => return,
    };

    let vocab = 32usize;
    let hidden = 256usize;
    let q8_table = build_dummy_q8_0_table(vocab, hidden);
    let hidden_vec: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.017).sin()).collect();

    let cpu_argmax = cpu_q8_0_matmul_argmax(&q8_table, &hidden_vec, vocab, hidden);
    let gpu_argmax = gemv
        .logit_argmax(
            &hidden_vec,
            &q8_table,
            QuantType::Q8_0,
            vocab as u32,
            hidden as u32,
        )
        .expect("logit_argmax q8_0");
    assert_eq!(gpu_argmax, cpu_argmax, "gpu={gpu_argmax} cpu={cpu_argmax}");
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

fn build_dummy_q8_0_table(vocab: usize, hidden: usize) -> Vec<u8> {
    assert_eq!(hidden % 32, 0);
    let blocks_per_row = hidden / 32;
    let row_bytes = blocks_per_row * 34;
    let mut bytes = vec![0u8; vocab * row_bytes];
    for v in 0..vocab {
        for b in 0..blocks_per_row {
            let off = v * row_bytes + b * 34;
            let d = half::f16::from_f32(0.01 + v as f32 * 0.0003 + b as f32 * 0.0007);
            bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
            for i in 0..32 {
                let q = ((v as i32 * 11 + b as i32 * 7 + i as i32 * 3) % 255 - 127) as i8;
                bytes[off + 2 + i] = q as u8;
            }
        }
    }
    bytes
}

fn cpu_q6k_matmul_argmax(table: &[u8], hidden_vec: &[f32], vocab: usize, hidden: usize) -> u32 {
    use rnb_cpu::quantize::blocks::BlockQ6_K;
    use rnb_cpu::quantize::dequant::dequantize_q6_k;
    use std::mem::size_of;
    assert_eq!(hidden % 256, 0);
    let blocks_per_row = hidden / 256;
    let row_bytes = blocks_per_row * size_of::<BlockQ6_K>();
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for v in 0..vocab {
        let mut sum = 0.0f32;
        let row_off = v * row_bytes;
        for b in 0..blocks_per_row {
            let block_off = row_off + b * size_of::<BlockQ6_K>();
            let block: BlockQ6_K = read_block(table, block_off / size_of::<BlockQ6_K>());
            let mut buf = [0.0f32; 256];
            dequantize_q6_k(&block, &mut buf);
            for (i, &x) in buf.iter().enumerate() {
                sum += x * hidden_vec[b * 256 + i];
            }
        }
        if sum > best_val {
            best_val = sum;
            best_idx = v as u32;
        }
    }
    best_idx
}

fn cpu_q8_0_matmul_argmax(table: &[u8], hidden_vec: &[f32], vocab: usize, hidden: usize) -> u32 {
    assert_eq!(hidden % 32, 0);
    let blocks_per_row = hidden / 32;
    let row_bytes = blocks_per_row * 34;
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for v in 0..vocab {
        let mut sum = 0.0f32;
        let row_off = v * row_bytes;
        for b in 0..blocks_per_row {
            let block_off = row_off + b * 34;
            let d = half::f16::from_le_bytes([table[block_off], table[block_off + 1]]).to_f32();
            for i in 0..32 {
                let q = table[block_off + 2 + i] as i8 as f32;
                sum += d * q * hidden_vec[b * 32 + i];
            }
        }
        if sum > best_val {
            best_val = sum;
            best_idx = v as u32;
        }
    }
    best_idx
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
