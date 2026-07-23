#[cfg(target_arch = "aarch64")]
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(target_arch = "aarch64")]
use half::f16;
#[cfg(target_arch = "aarch64")]
use rnb_cpu::gemm::quant_gemv::{gemv_quantized, QuantGemvType};

#[cfg(target_arch = "aarch64")]
fn quant_layout(quant: QuantGemvType) -> (usize, usize) {
    match quant {
        QuantGemvType::Q4_0 => (32, 18),
        QuantGemvType::Q4_1 => (32, 20),
        QuantGemvType::Q5_0 => (32, 22),
        QuantGemvType::Q5_1 => (32, 24),
        QuantGemvType::Q8_0 => (32, 34),
        QuantGemvType::Q8_1 => (32, 36),
        QuantGemvType::Q2K => (256, 84),
        QuantGemvType::Q3K => (256, 110),
        QuantGemvType::IQ2XXS => (256, 66),
        QuantGemvType::IQ2S => (256, 82),
        QuantGemvType::IQ3XXS => (256, 98),
        QuantGemvType::IQ4XS => (256, 136),
        _ => unreachable!("benchmark only covers quant-general kernels"),
    }
}

#[cfg(target_arch = "aarch64")]
fn seeded_weights(quant: QuantGemvType, rows: usize, cols: usize) -> (Vec<u8>, usize) {
    let (block_elems, block_bytes) = quant_layout(quant);
    let blocks_per_row = cols / block_elems;
    let bytes_per_row = blocks_per_row * block_bytes;
    let mut bytes = vec![0u8; rows * bytes_per_row];
    let mut state = 0x1234_5678u32;
    for byte in &mut bytes {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    let d = f16::from_f32(0.03125).to_bits().to_le_bytes();
    let min = f16::from_f32(0.0078125).to_bits().to_le_bytes();
    for block in bytes.chunks_exact_mut(block_bytes) {
        match quant {
            QuantGemvType::Q2K => {
                block[block_bytes - 4..block_bytes - 2].copy_from_slice(&d);
                block[block_bytes - 2..].copy_from_slice(&min);
            }
            QuantGemvType::Q3K => block[block_bytes - 2..].copy_from_slice(&d),
            QuantGemvType::Q4_1 | QuantGemvType::Q5_1 | QuantGemvType::Q8_1 => {
                block[..2].copy_from_slice(&d);
                block[2..4].copy_from_slice(&min);
            }
            _ => block[..2].copy_from_slice(&d),
        }
    }
    (bytes, bytes_per_row)
}

#[cfg(target_arch = "aarch64")]
fn bench_quant_gemv(c: &mut Criterion) {
    const ROWS: usize = 1024;
    const COLS: usize = 2048;
    let input = (0..COLS)
        .map(|index| ((index as f32 * 0.019).sin() * 2.75) + (index % 13) as f32 * 0.01)
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("quant_gemv");
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    for (label, quant) in [
        ("q4_0", QuantGemvType::Q4_0),
        ("q4_1", QuantGemvType::Q4_1),
        ("q5_0", QuantGemvType::Q5_0),
        ("q5_1", QuantGemvType::Q5_1),
        ("q8_0", QuantGemvType::Q8_0),
        ("q8_1", QuantGemvType::Q8_1),
        ("q2_k", QuantGemvType::Q2K),
        ("q3_k", QuantGemvType::Q3K),
        ("iq2_xxs", QuantGemvType::IQ2XXS),
        ("iq2_s", QuantGemvType::IQ2S),
        ("iq3_xxs", QuantGemvType::IQ3XXS),
        ("iq4_xs", QuantGemvType::IQ4XS),
    ] {
        let (weights, bytes_per_row) = seeded_weights(quant, ROWS, COLS);
        let mut output = vec![0.0f32; ROWS];
        group.bench_function(label, |b| {
            b.iter(|| {
                gemv_quantized(
                    black_box(&weights),
                    black_box(&input),
                    black_box(&mut output),
                    ROWS,
                    COLS,
                    1,
                    bytes_per_row,
                    quant,
                )
            });
        });
    }
    group.finish();
}

#[cfg(not(target_arch = "aarch64"))]
fn bench_quant_gemv(c: &mut Criterion) {
    c.bench_function("quant_gemv/aarch64_only", |b| b.iter(|| black_box(1 + 1)));
}

criterion_group!(benches, bench_quant_gemv);
criterion_main!(benches);
