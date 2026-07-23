use std::process::ExitCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmokeSuite {
    All,
    Gdn,
    Attention,
    Roundtrip,
}

fn parse_requested_suites(args: &[&str]) -> Result<Vec<SmokeSuite>, String> {
    if args.is_empty() {
        return Ok(vec![SmokeSuite::All]);
    }
    args.iter()
        .map(|arg| match *arg {
            "all" => Ok(SmokeSuite::All),
            "gdn" => Ok(SmokeSuite::Gdn),
            "attention" => Ok(SmokeSuite::Attention),
            "roundtrip" => Ok(SmokeSuite::Roundtrip),
            other => Err(format!(
                "unknown suite '{other}' (expected all, gdn, attention, or roundtrip)"
            )),
        })
        .collect()
}

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    let suites = match parse_requested_suites(&arg_refs) {
        Ok(suites) => suites,
        Err(e) => {
            eprintln!("{e}");
            eprintln!("usage: rnb-vulkan-smoke [all|gdn|attention|roundtrip]...");
            return ExitCode::from(2);
        }
    };

    match run_smoke_suites(&suites) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[vulkan-smoke] failed: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "vulkan")]
fn run_smoke_suites(suites: &[SmokeSuite]) -> Result<(), String> {
    let mut vk = rnb_backend_vulkan::VulkanLayerGemv::new(
        1024,
        248_320,
        64,
        rnb_backend_vulkan::GpuWeightMode::Soa,
    )?;
    let expanded = if suites.contains(&SmokeSuite::All) {
        vec![
            SmokeSuite::Gdn,
            SmokeSuite::Attention,
            SmokeSuite::Roundtrip,
        ]
    } else {
        suites.to_vec()
    };

    for suite in expanded {
        match suite {
            SmokeSuite::All => {}
            SmokeSuite::Gdn => run_gdn_smoke(&mut vk)?,
            SmokeSuite::Attention => run_attention_smoke(&mut vk)?,
            SmokeSuite::Roundtrip => run_roundtrip_smoke(&mut vk)?,
        }
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn run_smoke_suites(_suites: &[SmokeSuite]) -> Result<(), String> {
    Err("rnb-vulkan-smoke requires --features vulkan".into())
}

#[cfg(feature = "vulkan")]
fn run_gdn_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    eprintln!("[vulkan-smoke] gdn_delta_step");
    vk.self_test_gdn_delta_step()?;
    eprintln!("[vulkan-smoke] gdn_gated_norm_silu");
    vk.self_test_gdn_gated_norm_silu()?;
    eprintln!("[vulkan-smoke] gdn_qkv_conv_window");
    vk.self_test_gdn_qkv_conv_window()?;
    eprintln!("[vulkan-smoke] gdn_qkv_conv_window_resident_conv_state");
    vk.self_test_gdn_qkv_conv_window_resident_conv_state()?;
    eprintln!("[vulkan-smoke] gdn_qkv_conv_window_resident_conv_state_strided");
    vk.self_test_gdn_qkv_conv_window_resident_conv_state_strided()?;
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_attention_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    eprintln!("[vulkan-smoke] q_window_into_kv_mirror_and_decode_grouped");
    vk.self_test_q_window_into_kv_mirror_and_decode_grouped()?;
    eprintln!("[vulkan-smoke] q_window_into_kv_mirror_and_decode_grouped_combined");
    vk.self_test_q_window_into_kv_mirror_and_decode_grouped_combined()?;
    eprintln!("[vulkan-smoke] gated_q_norm_rope_chain");
    vk.self_test_gated_q_norm_rope_chain()?;
    eprintln!("[vulkan-smoke] attention_decode_window");
    vk.self_test_attention_decode_window()?;
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_roundtrip_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    eprintln!("[vulkan-smoke] q8_0_gemv_nonzero");
    vk.self_test_q8_0_gemv_nonzero()?;
    eprintln!("[vulkan-smoke] q8_0_gemv_multiblock_matches_cpu");
    vk.self_test_q8_0_gemv_multiblock_matches_cpu()?;
    eprintln!("[vulkan-smoke] q5k_gemv_matches_cpu");
    run_q5k_gemv_smoke(vk)?;
    eprintln!("[vulkan-smoke] q6k_gemv_matches_cpu");
    run_q6k_gemv_smoke(vk)?;
    eprintln!("[vulkan-smoke] quantized_embedding_matches_cpu");
    run_quantized_embedding_smoke(vk)?;
    eprintln!("[vulkan-smoke] q5k_q8k_matches_cpu");
    let q5k_q8k_diff = vk.self_test_q5k_q8k()?;
    if q5k_q8k_diff >= 0.05 {
        return Err(format!("q5k_q8k max_diff too high: {q5k_q8k_diff}"));
    }
    eprintln!("[vulkan-smoke] q8_q8k_matches_cpu");
    let q8_q8k_diff = vk.self_test_q8_q8k()?;
    if q8_q8k_diff >= 0.05 {
        return Err(format!("q8_q8k max_diff too high: {q8_q8k_diff}"));
    }
    eprintln!("[vulkan-smoke] native_quant_projection_matches_cpu");
    run_native_quant_projection_smoke(vk)?;
    eprintln!("[vulkan-smoke] q4k_block_parallel");
    let q4k_diff = vk.self_test_q4k_block_parallel()?;
    if q4k_diff >= 0.05 {
        return Err(format!("q4k_block_parallel max_diff too high: {q4k_diff}"));
    }
    eprintln!("[vulkan-smoke] q4k_logit_argmax_matches_cpu");
    run_q4k_logit_argmax_smoke(vk)?;
    eprintln!("[vulkan-smoke] q5k_logit_argmax_matches_cpu");
    run_q5k_logit_argmax_smoke(vk)?;
    eprintln!("[vulkan-smoke] argmax_pairs_f32_large_count");
    vk.self_test_argmax_pairs_f32_large_count()?;
    eprintln!("[vulkan-smoke] prefill_hidden_roundtrip");
    vk.self_test_prefill_hidden_roundtrip()?;
    eprintln!("[vulkan-smoke] prefill_hidden_offset_writes");
    vk.self_test_prefill_hidden_offset_writes()?;
    eprintln!("[vulkan-smoke] q_window_into_kv_mirror_avoids_kv_host_roundtrip");
    vk.self_test_q_window_into_kv_mirror_avoids_kv_host_roundtrip()?;
    eprintln!("[vulkan-smoke] q_window_decode_project_avoids_attn_host_roundtrip");
    vk.self_test_q_window_decode_project_avoids_attn_host_roundtrip()?;
    eprintln!("[vulkan-smoke] q_window_decode_project_elides_q_host_download");
    vk.self_test_q_window_decode_project_elides_q_host_download()?;
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_q5k_gemv_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    use rnb_backend_vulkan::{QuantType, WeightId, WeightKind};

    const ROWS: usize = 3;
    const COLS: usize = 512;
    const BLOCKS_PER_ROW: usize = COLS / 256;
    const BLOCK_BYTES: usize = 176;

    let input: Vec<f32> = (0..COLS)
        .map(|i| (((i * 37 + 11) % 127) as f32 - 63.0) * 0.00390625)
        .collect();
    let mut raw = vec![0u8; ROWS * BLOCKS_PER_ROW * BLOCK_BYTES];
    let mut expected = vec![0.0f32; ROWS];

    for row in 0..ROWS {
        for blk in 0..BLOCKS_PER_ROW {
            let block_offset = (row * BLOCKS_PER_ROW + blk) * BLOCK_BYTES;
            let block = &mut raw[block_offset..block_offset + BLOCK_BYTES];
            let d = half::f16::from_f32(0.001953125 * ((row + 1) * (blk + 1)) as f32);
            let dmin = half::f16::from_f32(0.0009765625 * (row + blk + 1) as f32);
            block[..2].copy_from_slice(&d.to_le_bytes());
            block[2..4].copy_from_slice(&dmin.to_le_bytes());
            for (i, value) in block[4..16].iter_mut().enumerate() {
                *value = (i * 29 + row * 17 + blk * 11) as u8;
            }
            for (i, value) in block[16..48].iter_mut().enumerate() {
                *value = (i * 13 + row * 19 + blk * 23) as u8;
            }
            for (i, value) in block[48..176].iter_mut().enumerate() {
                let low = ((i * 7 + row * 3 + blk) % 16) as u8;
                let high = ((i * 11 + row * 5 + blk * 2) % 16) as u8;
                *value = low | (high << 4);
            }

            let mut scales = [0u8; 8];
            let mut mins = [0u8; 8];
            for index in 0..8 {
                (scales[index], mins[index]) = if index < 4 {
                    (block[4 + index] & 63, block[8 + index] & 63)
                } else {
                    (
                        (block[8 + index] & 0x0f) | ((block[index] >> 6) << 4),
                        (block[8 + index] >> 4) | ((block[4 + index] >> 6) << 4),
                    )
                };
            }

            let d = d.to_f32();
            let dmin = dmin.to_f32();
            for group in 0..4 {
                let qh_mask_low = 1u8 << (group * 2);
                let qh_mask_high = 2u8 << (group * 2);
                for lane in 0..32 {
                    let packed = block[48 + group * 32 + lane];
                    let qh = block[16 + lane];
                    let low = (packed & 0x0f) + if qh & qh_mask_low != 0 { 16 } else { 0 };
                    let high = (packed >> 4) + if qh & qh_mask_high != 0 { 16 } else { 0 };
                    let low_element = group * 64 + lane;
                    let high_element = low_element + 32;
                    expected[row] += (d * scales[group * 2] as f32 * low as f32
                        - dmin * mins[group * 2] as f32)
                        * input[blk * 256 + low_element];
                    expected[row] += (d * scales[group * 2 + 1] as f32 * high as f32
                        - dmin * mins[group * 2 + 1] as f32)
                        * input[blk * 256 + high_element];
                }
            }
        }
    }

    let mut got = vec![0.0f32; ROWS];
    vk.gemv(
        WeightId {
            layer: u16::MAX,
            kind: WeightKind::GdnGate,
        },
        &raw,
        ROWS,
        COLS,
        QuantType::Q5K,
        &input,
        &mut got,
    )?;

    let max_diff = got
        .iter()
        .zip(&expected)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    let tolerance = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(1.0f32, f32::max)
        * 1e-3;
    if max_diff > tolerance {
        return Err(format!(
            "Q5_K GEMV mismatch: max_diff={max_diff} tolerance={tolerance} got={got:?} expected={expected:?}"
        ));
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_q6k_gemv_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    use rnb_backend_vulkan::{QuantType, WeightId, WeightKind};

    const ROWS: usize = 3;
    const COLS: usize = 512;
    const BLOCKS_PER_ROW: usize = COLS / 256;
    const BLOCK_BYTES: usize = 210;

    let input: Vec<f32> = (0..COLS)
        .map(|i| (((i * 37 + 11) % 127) as f32 - 63.0) * 0.00390625)
        .collect();
    let mut raw = vec![0u8; ROWS * BLOCKS_PER_ROW * BLOCK_BYTES];
    let mut expected = vec![0.0f32; ROWS];

    for row in 0..ROWS {
        for blk in 0..BLOCKS_PER_ROW {
            let block_offset = (row * BLOCKS_PER_ROW + blk) * BLOCK_BYTES;
            let block = &mut raw[block_offset..block_offset + BLOCK_BYTES];
            for (i, value) in block[..128].iter_mut().enumerate() {
                *value = (i * 17 + row * 11 + blk * 7) as u8;
            }
            for (i, value) in block[128..192].iter_mut().enumerate() {
                *value = (i * 13 + row * 19 + blk * 23) as u8;
            }
            for (i, value) in block[192..208].iter_mut().enumerate() {
                let scale = ((i * 5 + row * 3 + blk) % 31) as i8 - 15;
                *value = if scale == 0 { 1 } else { scale } as u8;
            }
            let d = half::f16::from_f32(0.001953125 * ((row + 1) * (blk + 1)) as f32);
            block[208..210].copy_from_slice(&d.to_le_bytes());

            let ql = &block[..128];
            let qh = &block[128..192];
            let scales = &block[192..208];
            let d = d.to_f32();
            for group in 0..2 {
                let ql_base = group * 64;
                let qh_base = group * 32;
                let scale_base = group * 8;
                let element_base = group * 128;
                for lane in 0..32 {
                    let scale_lane = lane / 16;
                    let quants = [
                        (ql[ql_base + lane] & 0x0f) | (((qh[qh_base + lane] >> 0) & 3) << 4),
                        (ql[ql_base + lane + 32] & 0x0f) | (((qh[qh_base + lane] >> 2) & 3) << 4),
                        (ql[ql_base + lane] >> 4) | (((qh[qh_base + lane] >> 4) & 3) << 4),
                        (ql[ql_base + lane + 32] >> 4) | (((qh[qh_base + lane] >> 6) & 3) << 4),
                    ];
                    for (stripe, quant) in quants.into_iter().enumerate() {
                        let element = element_base + lane + stripe * 32;
                        let scale = scales[scale_base + scale_lane + stripe * 2] as i8 as f32;
                        expected[row] +=
                            d * scale * (quant as i32 - 32) as f32 * input[blk * 256 + element];
                    }
                }
            }
        }
    }

    let mut got = vec![0.0f32; ROWS];
    vk.gemv(
        WeightId {
            layer: u16::MAX,
            kind: WeightKind::GdnQkv,
        },
        &raw,
        ROWS,
        COLS,
        QuantType::Q6K,
        &input,
        &mut got,
    )?;

    let max_diff = got
        .iter()
        .zip(&expected)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    let tolerance = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(1.0f32, f32::max)
        * 1e-3;
    if max_diff > tolerance {
        return Err(format!(
            "Q6_K GEMV mismatch: max_diff={max_diff} tolerance={tolerance} got={got:?} expected={expected:?}"
        ));
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_native_quant_case(
    vk: &mut rnb_backend_vulkan::VulkanLayerGemv,
    layer: u16,
    label: &str,
    quant: rnb_backend_vulkan::QuantType,
    row_bytes: &[u8],
    decoded: &[f32],
) -> Result<(), String> {
    use rnb_backend_vulkan::{WeightId, WeightKind};

    let input = (0..decoded.len())
        .map(|index| ((index as f32 * 0.173).sin() * 0.9) + ((index as f32 * 0.071).cos() * 0.35))
        .collect::<Vec<_>>();
    let expected = decoded
        .iter()
        .zip(&input)
        .map(|(weight, activation)| weight * activation)
        .sum::<f32>();
    let mut raw = Vec::with_capacity(row_bytes.len() * 2);
    raw.extend_from_slice(row_bytes);
    raw.extend_from_slice(row_bytes);
    let mut window_input = Vec::with_capacity(input.len() * 2);
    window_input.extend_from_slice(&input);
    window_input.extend_from_slice(&input);
    let mut window_got = [0.0f32; 4];
    vk.gemv_window(
        WeightId {
            layer,
            kind: WeightKind::QProj,
        },
        &raw,
        2,
        decoded.len(),
        quant,
        &window_input,
        &mut window_got,
    )?;
    let window_max_diff = window_got
        .iter()
        .map(|actual| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let tolerance = expected.abs().max(1.0) * 2e-4;
    if window_max_diff > tolerance {
        return Err(format!(
            "{label} native window GEMV mismatch: max_diff={window_max_diff} tolerance={tolerance} got={window_got:?} expected={expected}"
        ));
    }
    let mut window_batch_a = [0.0f32; 4];
    let mut window_batch_b = [0.0f32; 4];
    vk.gemv_multi_window(
        &window_input,
        decoded.len(),
        &[
            (
                WeightId {
                    layer,
                    kind: WeightKind::KProj,
                },
                raw.as_slice(),
                2,
                decoded.len(),
                quant,
            ),
            (
                WeightId {
                    layer,
                    kind: WeightKind::VProj,
                },
                raw.as_slice(),
                2,
                decoded.len(),
                quant,
            ),
        ],
        &mut [&mut window_batch_a, &mut window_batch_b],
    )?;
    let window_batch_max_diff = window_batch_a
        .iter()
        .chain(&window_batch_b)
        .map(|actual| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if window_batch_max_diff > tolerance {
        return Err(format!(
            "{label} native batched window GEMV mismatch: max_diff={window_batch_max_diff} tolerance={tolerance} got={window_batch_a:?}/{window_batch_b:?} expected={expected}"
        ));
    }
    let mut batch_a = [0.0f32; 2];
    let mut batch_b = [0.0f32; 2];
    vk.gemv_multi(
        &input,
        &[
            (
                WeightId {
                    layer,
                    kind: WeightKind::KProj,
                },
                raw.as_slice(),
                2,
                decoded.len(),
                quant,
            ),
            (
                WeightId {
                    layer,
                    kind: WeightKind::VProj,
                },
                raw.as_slice(),
                2,
                decoded.len(),
                quant,
            ),
        ],
        &mut [&mut batch_a, &mut batch_b],
    )?;
    let batch_max_diff = batch_a
        .iter()
        .chain(&batch_b)
        .map(|actual| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if batch_max_diff > tolerance {
        return Err(format!(
            "{label} native batched GEMV mismatch: max_diff={batch_max_diff} tolerance={tolerance} got={batch_a:?}/{batch_b:?} expected={expected}"
        ));
    }
    let mut got = [0.0f32; 2];
    vk.gemv(
        WeightId {
            layer,
            kind: WeightKind::QProj,
        },
        &raw,
        2,
        decoded.len(),
        quant,
        &input,
        &mut got,
    )?;
    let max_diff = got
        .iter()
        .map(|actual| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let tolerance = expected.abs().max(1.0) * 2e-4;
    if max_diff > tolerance {
        return Err(format!(
            "{label} native GEMV mismatch: max_diff={max_diff} tolerance={tolerance} got={got:?} expected={expected}"
        ));
    }
    let winner = vk.logit_argmax(&input, &raw, quant, 2, decoded.len() as u32)?;
    if winner != 0 {
        return Err(format!(
            "{label} native logit argmax mismatch: got={winner} expected=0"
        ));
    }
    let mut embedding = vec![0.0f32; decoded.len() * 2];
    vk.embed_lookup(
        &[1, 0],
        decoded.len() as u32,
        2,
        quant,
        &raw,
        &mut embedding,
    )?;
    let embedding_diff = embedding
        .chunks_exact(decoded.len())
        .flat_map(|row| row.iter().zip(decoded))
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    if embedding_diff > 1e-5 {
        return Err(format!(
            "{label} native embedding mismatch: max_diff={embedding_diff}"
        ));
    }
    eprintln!("[vulkan-smoke] native_quant_case={label}");
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_native_quant_projection_smoke(
    vk: &mut rnb_backend_vulkan::VulkanLayerGemv,
) -> Result<(), String> {
    use half::{bf16, f16};
    use rnb_backend_vulkan::QuantType;
    use rnb_cpu::gemm::dequant::{dequantize_bytes_to_f32, DequantType};
    use rnb_cpu::quantize::{
        dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q5_0,
        dequantize_q5_1, BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1,
        BlockQ8_1,
    };

    fn ggml_reference_block(len: usize, has_leading_f16: bool) -> Vec<u8> {
        let mut block = (0..len)
            .map(|index| (index.wrapping_mul(73).wrapping_add(19) & 0xff) as u8)
            .collect::<Vec<_>>();
        if has_leading_f16 {
            block[..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        }
        block
    }

    let f32_values = (0..32)
        .map(|index| (index as f32 - 15.0) * 0.03125)
        .collect::<Vec<_>>();
    let mut f32_bytes = Vec::with_capacity(f32_values.len() * 4);
    for value in &f32_values {
        f32_bytes.extend_from_slice(&value.to_le_bytes());
    }
    run_native_quant_case(vk, 500, "F32", QuantType::F32, &f32_bytes, &f32_values)?;

    let mut f16_bytes = Vec::with_capacity(f32_values.len() * 2);
    let mut f16_values = Vec::with_capacity(f32_values.len());
    for value in &f32_values {
        let stored = f16::from_f32(*value);
        f16_bytes.extend_from_slice(&stored.to_le_bytes());
        f16_values.push(stored.to_f32());
    }
    run_native_quant_case(vk, 501, "F16", QuantType::F16, &f16_bytes, &f16_values)?;

    let mut bf16_bytes = Vec::with_capacity(f32_values.len() * 2);
    let mut bf16_values = Vec::with_capacity(f32_values.len());
    for value in &f32_values {
        let stored = bf16::from_f32(*value);
        bf16_bytes.extend_from_slice(&stored.to_le_bytes());
        bf16_values.push(stored.to_f32());
    }
    run_native_quant_case(vk, 502, "BF16", QuantType::BF16, &bf16_bytes, &bf16_values)?;

    let q4_0 = BlockQ4_0 {
        d: f16::from_f32(0.075),
        qs: std::array::from_fn(|index| {
            let lo = ((index * 5 + 3) & 0x0f) as u8;
            let hi = ((index * 11 + 7) & 0x0f) as u8;
            lo | (hi << 4)
        }),
    };
    let mut q4_0_values = [0.0f32; 32];
    dequantize_q4_0(&q4_0, &mut q4_0_values);
    let mut q4_0_bytes = Vec::new();
    append_block_bytes(&mut q4_0_bytes, &q4_0);
    run_native_quant_case(vk, 503, "Q4_0", QuantType::Q4_0, &q4_0_bytes, &q4_0_values)?;

    let q4_1 = BlockQ4_1 {
        d: f16::from_f32(0.0625),
        m: f16::from_f32(-0.4),
        qs: std::array::from_fn(|index| {
            let lo = ((index * 7 + 1) & 0x0f) as u8;
            let hi = ((index * 13 + 9) & 0x0f) as u8;
            lo | (hi << 4)
        }),
    };
    let mut q4_1_values = [0.0f32; 32];
    dequantize_q4_1(&q4_1, &mut q4_1_values);
    let mut q4_1_bytes = Vec::new();
    append_block_bytes(&mut q4_1_bytes, &q4_1);
    run_native_quant_case(vk, 504, "Q4_1", QuantType::Q4_1, &q4_1_bytes, &q4_1_values)?;

    let q5_0 = BlockQ5_0 {
        d: f16::from_f32(0.04),
        qh: [0x96, 0x3c, 0xa5, 0x69],
        qs: std::array::from_fn(|index| {
            let lo = ((index * 3 + 2) & 0x0f) as u8;
            let hi = ((index * 9 + 5) & 0x0f) as u8;
            lo | (hi << 4)
        }),
    };
    let mut q5_0_values = [0.0f32; 32];
    dequantize_q5_0(&q5_0, &mut q5_0_values);
    let mut q5_0_bytes = Vec::new();
    append_block_bytes(&mut q5_0_bytes, &q5_0);
    run_native_quant_case(vk, 505, "Q5_0", QuantType::Q5_0, &q5_0_bytes, &q5_0_values)?;

    let q5_1 = BlockQ5_1 {
        d: f16::from_f32(0.035),
        m: f16::from_f32(-0.25),
        qh: [0x5a, 0xc3, 0x78, 0x0f],
        qs: std::array::from_fn(|index| {
            let lo = ((index * 11 + 4) & 0x0f) as u8;
            let hi = ((index * 5 + 12) & 0x0f) as u8;
            lo | (hi << 4)
        }),
    };
    let mut q5_1_values = [0.0f32; 32];
    dequantize_q5_1(&q5_1, &mut q5_1_values);
    let mut q5_1_bytes = Vec::new();
    append_block_bytes(&mut q5_1_bytes, &q5_1);
    run_native_quant_case(vk, 506, "Q5_1", QuantType::Q5_1, &q5_1_bytes, &q5_1_values)?;

    let q8_1 = BlockQ8_1 {
        d: f16::from_f32(0.0125),
        s: f16::from_f32(0.0),
        qs: std::array::from_fn(|index| ((index * 23 + 19) % 255) as i16 as i8 - 127i8),
    };
    let q8_1_values = q8_1
        .qs
        .iter()
        .map(|value| *value as f32 * q8_1.d.to_f32())
        .collect::<Vec<_>>();
    let mut q8_1_bytes = Vec::new();
    append_block_bytes(&mut q8_1_bytes, &q8_1);
    run_native_quant_case(vk, 507, "Q8_1", QuantType::Q8_1, &q8_1_bytes, &q8_1_values)?;

    let q2k = BlockQ2_K {
        scales: std::array::from_fn(|index| {
            (((index * 5 + 3) & 0x0f) | (((index * 7 + 1) & 0x0f) << 4)) as u8
        }),
        qs: std::array::from_fn(|index| ((index * 37 + 11) & 0xff) as u8),
        d: f16::from_f32(0.03125),
        dmin: f16::from_f32(0.015625),
    };
    let mut q2k_values = [0.0f32; 256];
    dequantize_q2_k(&q2k, &mut q2k_values);
    let mut q2k_bytes = Vec::new();
    append_block_bytes(&mut q2k_bytes, &q2k);
    run_native_quant_case(vk, 508, "Q2_K", QuantType::Q2K, &q2k_bytes, &q2k_values)?;

    let q3k = BlockQ3_K {
        hmask: std::array::from_fn(|index| ((index * 29 + 17) & 0xff) as u8),
        qs: std::array::from_fn(|index| ((index * 41 + 7) & 0xff) as u8),
        scales: std::array::from_fn(|index| ((index * 53 + 13) & 0xff) as u8),
        d: f16::from_f32(0.01875),
    };
    let mut q3k_values = [0.0f32; 256];
    dequantize_q3_k(&q3k, &mut q3k_values);
    let mut q3k_bytes = Vec::new();
    append_block_bytes(&mut q3k_bytes, &q3k);
    run_native_quant_case(vk, 509, "Q3_K", QuantType::Q3K, &q3k_bytes, &q3k_values)?;

    let mut q8k = vec![0u8; 292];
    q8k[..4].copy_from_slice(&0.0125f32.to_le_bytes());
    for (index, value) in q8k[4..260].iter_mut().enumerate() {
        *value = ((index * 37 + 11) & 0xff) as u8;
    }
    let q8k_values = dequantize_bytes_to_f32(&q8k, DequantType::Q8K);
    run_native_quant_case(vk, 510, "Q8_K", QuantType::Q8K, &q8k, &q8k_values)?;

    let mut iq4_nl = vec![0u8; 18];
    iq4_nl[..2].copy_from_slice(&f16::from_f32(0.00625).to_le_bytes());
    for (index, value) in iq4_nl[2..].iter_mut().enumerate() {
        *value = ((index * 29 + 7) & 0xff) as u8;
    }
    let iq4_nl_values = dequantize_bytes_to_f32(&iq4_nl, DequantType::IQ4NL);
    run_native_quant_case(
        vk,
        511,
        "IQ4_NL",
        QuantType::IQ4_NL,
        &iq4_nl,
        &iq4_nl_values,
    )?;

    let mut iq4_xs = vec![0u8; 136];
    iq4_xs[..2].copy_from_slice(&f16::from_f32(0.004).to_le_bytes());
    iq4_xs[2..4].copy_from_slice(&0x5a39u16.to_le_bytes());
    iq4_xs[4..8].copy_from_slice(&[0x94, 0x2d, 0xe1, 0x6b]);
    for (index, value) in iq4_xs[8..].iter_mut().enumerate() {
        *value = ((index * 43 + 19) & 0xff) as u8;
    }
    let iq4_xs_values = dequantize_bytes_to_f32(&iq4_xs, DequantType::IQ4XS);
    run_native_quant_case(
        vk,
        512,
        "IQ4_XS",
        QuantType::IQ4_XS,
        &iq4_xs,
        &iq4_xs_values,
    )?;

    for (offset, (label, quant, dequant, block_bytes, has_leading_f16)) in [
        ("IQ2_XXS", QuantType::IQ2_XXS, DequantType::IQ2XXS, 66, true),
        ("IQ2_XS", QuantType::IQ2_XS, DequantType::IQ2XS, 74, true),
        ("IQ2_S", QuantType::IQ2_S, DequantType::IQ2S, 82, true),
        ("IQ3_XXS", QuantType::IQ3_XXS, DequantType::IQ3XXS, 98, true),
        ("IQ3_S", QuantType::IQ3_S, DequantType::IQ3S, 110, true),
        ("IQ1_S", QuantType::IQ1_S, DequantType::IQ1S, 50, true),
        ("IQ1_M", QuantType::IQ1_M, DequantType::IQ1M, 56, false),
    ]
    .into_iter()
    .enumerate()
    {
        let row_bytes = ggml_reference_block(block_bytes, has_leading_f16);
        let decoded = dequantize_bytes_to_f32(&row_bytes, dequant);
        run_native_quant_case(vk, 519 + offset as u16, label, quant, &row_bytes, &decoded)?;
    }

    let mut tq1 = vec![0u8; 54];
    for (index, value) in tq1[..52].iter_mut().enumerate() {
        *value = ((index * 47 + 23) % 243) as u8;
    }
    tq1[52..54].copy_from_slice(&f16::from_f32(0.05).to_le_bytes());
    let tq1_values = dequantize_bytes_to_f32(&tq1, DequantType::TQ1_0);
    run_native_quant_case(vk, 513, "TQ1_0", QuantType::TQ1_0, &tq1, &tq1_values)?;

    let mut tq2 = vec![0u8; 66];
    for (index, value) in tq2[..64].iter_mut().enumerate() {
        *value = ((index * 31 + 17) & 0xff) as u8;
    }
    tq2[64..66].copy_from_slice(&f16::from_f32(0.04).to_le_bytes());
    let tq2_values = dequantize_bytes_to_f32(&tq2, DequantType::TQ2_0);
    run_native_quant_case(vk, 514, "TQ2_0", QuantType::TQ2_0, &tq2, &tq2_values)?;

    let mut mxfp4 = vec![0u8; 17];
    mxfp4[0] = 124;
    for (index, value) in mxfp4[1..].iter_mut().enumerate() {
        *value = ((index * 59 + 13) & 0xff) as u8;
    }
    let mxfp4_values = dequantize_bytes_to_f32(&mxfp4, DequantType::MXFP4);
    run_native_quant_case(vk, 515, "MXFP4", QuantType::MXFP4, &mxfp4, &mxfp4_values)?;

    let mut nvfp4 = vec![0u8; 36];
    nvfp4[..4].copy_from_slice(&[0x30, 0x38, 0x40, 0x48]);
    for (index, value) in nvfp4[4..].iter_mut().enumerate() {
        *value = ((index * 61 + 29) & 0xff) as u8;
    }
    let nvfp4_values = dequantize_bytes_to_f32(&nvfp4, DequantType::NVFP4);
    run_native_quant_case(vk, 516, "NVFP4", QuantType::NVFP4, &nvfp4, &nvfp4_values)?;

    let mut q1_0 = vec![0u8; 18];
    q1_0[..2].copy_from_slice(&f16::from_f32(0.125).to_le_bytes());
    for (index, value) in q1_0[2..].iter_mut().enumerate() {
        *value = ((index * 67 + 31) & 0xff) as u8;
    }
    let q1_0_values = dequantize_bytes_to_f32(&q1_0, DequantType::Q1_0);
    run_native_quant_case(vk, 517, "Q1_0", QuantType::Q1_0, &q1_0, &q1_0_values)?;

    let mut q2_0 = vec![0u8; 18];
    q2_0[..2].copy_from_slice(&f16::from_f32(0.0625).to_le_bytes());
    for (index, value) in q2_0[2..].iter_mut().enumerate() {
        *value = ((index * 71 + 37) & 0xff) as u8;
    }
    let q2_0_values = dequantize_bytes_to_f32(&q2_0, DequantType::Q2_0);
    run_native_quant_case(vk, 518, "Q2_0", QuantType::Q2_0, &q2_0, &q2_0_values)?;

    Ok(())
}

#[cfg(feature = "vulkan")]
fn append_block_bytes<T>(out: &mut Vec<u8>, block: &T) {
    let bytes = unsafe {
        std::slice::from_raw_parts((block as *const T).cast::<u8>(), std::mem::size_of::<T>())
    };
    out.extend_from_slice(bytes);
}

#[cfg(feature = "vulkan")]
fn check_embedding_result(label: &str, got: &[f32], expected: &[f32]) -> Result<(), String> {
    let max_diff = got
        .iter()
        .zip(expected)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    if max_diff > 1e-5 {
        return Err(format!("{label} embedding mismatch: max_diff={max_diff}"));
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_quantized_embedding_smoke(
    vk: &mut rnb_backend_vulkan::VulkanLayerGemv,
) -> Result<(), String> {
    use rnb_backend_vulkan::QuantType;
    use rnb_cpu::quantize::blocks::{BlockQ4_K, BlockQ5_K, BlockQ8_0};
    use rnb_cpu::quantize::dequant::{dequantize_q4_k, dequantize_q5_k, dequantize_q8_0};

    const HIDDEN: usize = 256;
    const VOCAB: usize = 3;
    let token_ids = [2u32, 0];

    let mut q4_table = Vec::with_capacity(VOCAB * std::mem::size_of::<BlockQ4_K>());
    let mut q4_rows = Vec::with_capacity(VOCAB);
    for row in 0..VOCAB {
        let mut scales = [0u8; 12];
        for (index, value) in scales.iter_mut().enumerate() {
            *value = ((index * 31 + row * 19 + 7) & 0xff) as u8;
        }
        let mut qs = [0u8; 128];
        for (index, value) in qs.iter_mut().enumerate() {
            let low = ((index * 5 + row * 3) & 0x0f) as u8;
            let high = ((index * 11 + row * 7 + 1) & 0x0f) as u8;
            *value = low | (high << 4);
        }
        let block = BlockQ4_K {
            d: half::f16::from_f32(0.001953125 * (row + 1) as f32),
            dmin: half::f16::from_f32(0.0009765625 * (row + 2) as f32),
            scales,
            qs,
        };
        let mut values = [0.0f32; HIDDEN];
        dequantize_q4_k(&block, &mut values);
        q4_rows.push(values);
        append_block_bytes(&mut q4_table, &block);
    }
    let mut q4_expected = Vec::with_capacity(token_ids.len() * HIDDEN);
    for &token_id in &token_ids {
        q4_expected.extend_from_slice(&q4_rows[token_id as usize]);
    }
    let mut q4_got = vec![0.0f32; q4_expected.len()];
    vk.embed_lookup(
        &token_ids,
        HIDDEN as u32,
        VOCAB as u32,
        QuantType::Q4K,
        &q4_table,
        &mut q4_got,
    )?;
    check_embedding_result("Q4_K", &q4_got, &q4_expected)?;

    let mut q5_table = Vec::with_capacity(VOCAB * std::mem::size_of::<BlockQ5_K>());
    let mut q5_rows = Vec::with_capacity(VOCAB);
    for row in 0..VOCAB {
        let mut scales = [0u8; 12];
        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        for (index, value) in scales.iter_mut().enumerate() {
            *value = ((index * 23 + row * 29 + 5) & 0xff) as u8;
        }
        for (index, value) in qh.iter_mut().enumerate() {
            *value = ((index * 37 + row * 41 + 3) & 0xff) as u8;
        }
        for (index, value) in qs.iter_mut().enumerate() {
            let low = ((index * 7 + row * 5) & 0x0f) as u8;
            let high = ((index * 13 + row * 11 + 2) & 0x0f) as u8;
            *value = low | (high << 4);
        }
        let block = BlockQ5_K {
            d: half::f16::from_f32(0.0009765625 * (row + 1) as f32),
            dmin: half::f16::from_f32(0.00048828125 * (row + 2) as f32),
            scales,
            qh,
            qs,
        };
        let mut values = [0.0f32; HIDDEN];
        dequantize_q5_k(&block, &mut values);
        q5_rows.push(values);
        append_block_bytes(&mut q5_table, &block);
    }
    let mut q5_expected = Vec::with_capacity(token_ids.len() * HIDDEN);
    for &token_id in &token_ids {
        q5_expected.extend_from_slice(&q5_rows[token_id as usize]);
    }
    let mut q5_got = vec![0.0f32; q5_expected.len()];
    vk.embed_lookup(
        &token_ids,
        HIDDEN as u32,
        VOCAB as u32,
        QuantType::Q5K,
        &q5_table,
        &mut q5_got,
    )?;
    check_embedding_result("Q5_K", &q5_got, &q5_expected)?;

    let mut q8_table = Vec::with_capacity(VOCAB * HIDDEN / 32 * 34);
    let mut q8_rows = Vec::with_capacity(VOCAB);
    for row in 0..VOCAB {
        let mut values = [0.0f32; HIDDEN];
        for block_index in 0..(HIDDEN / 32) {
            let mut qs = [0i8; 32];
            for (index, value) in qs.iter_mut().enumerate() {
                *value = (((index * 17 + block_index * 13 + row * 19) % 255) as i32 - 127) as i8;
            }
            let block = BlockQ8_0 {
                d: half::f16::from_f32(0.00390625 * (row + block_index + 1) as f32),
                qs,
            };
            let mut dequantized = [0.0f32; 32];
            dequantize_q8_0(&block, &mut dequantized);
            values[block_index * 32..(block_index + 1) * 32].copy_from_slice(&dequantized);
            append_block_bytes(&mut q8_table, &block);
        }
        q8_rows.push(values);
    }
    let mut q8_expected = Vec::with_capacity(token_ids.len() * HIDDEN);
    for &token_id in &token_ids {
        q8_expected.extend_from_slice(&q8_rows[token_id as usize]);
    }
    let mut q8_got = vec![0.0f32; q8_expected.len()];
    vk.embed_lookup(
        &token_ids,
        HIDDEN as u32,
        VOCAB as u32,
        QuantType::Q8_0,
        &q8_table,
        &mut q8_got,
    )?;
    check_embedding_result("Q8_0", &q8_got, &q8_expected)
}

#[cfg(feature = "vulkan")]
fn run_q4k_logit_argmax_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    use rnb_backend_vulkan::QuantType;

    const VOCAB: usize = 17;
    const HIDDEN: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let hidden: Vec<f32> = (0..HIDDEN)
        .map(|i| (((i * 37 + 11) % 127) as f32 - 63.0) * 0.00390625)
        .collect();
    let mut table = vec![0u8; VOCAB * BLOCK_BYTES];
    let mut logits = vec![0.0f32; VOCAB];

    for row in 0..VOCAB {
        let block = &mut table[row * BLOCK_BYTES..(row + 1) * BLOCK_BYTES];
        let d = half::f16::from_f32(0.001953125 * (row % 7 + 1) as f32);
        let dmin = half::f16::from_f32(0.0009765625 * (row % 5 + 1) as f32);
        block[..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        for (i, value) in block[4..16].iter_mut().enumerate() {
            *value = (i * 29 + row * 17) as u8;
        }
        for (i, value) in block[16..].iter_mut().enumerate() {
            let low = ((i * 7 + row * 3) % 16) as u8;
            let high = ((i * 11 + row * 5 + 1) % 16) as u8;
            *value = low | (high << 4);
        }

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for index in 0..8 {
            (scales[index], mins[index]) = if index < 4 {
                (block[4 + index] & 63, block[8 + index] & 63)
            } else {
                (
                    (block[8 + index] & 0x0f) | ((block[index] >> 6) << 4),
                    (block[8 + index] >> 4) | ((block[4 + index] >> 6) << 4),
                )
            };
        }

        let d = d.to_f32();
        let dmin = dmin.to_f32();
        for group in 0..4 {
            for lane in 0..32 {
                let packed = block[16 + group * 32 + lane];
                let low_element = group * 64 + lane;
                let high_element = low_element + 32;
                logits[row] += (d * scales[group * 2] as f32 * (packed & 0x0f) as f32
                    - dmin * mins[group * 2] as f32)
                    * hidden[low_element];
                logits[row] += (d * scales[group * 2 + 1] as f32 * (packed >> 4) as f32
                    - dmin * mins[group * 2 + 1] as f32)
                    * hidden[high_element];
            }
        }
    }

    let expected = logits
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index as u32)
        .expect("VOCAB is nonzero");
    let got = vk.logit_argmax(&hidden, &table, QuantType::Q4K, VOCAB as u32, HIDDEN as u32)?;
    if got != expected {
        return Err(format!(
            "Q4_K logit argmax mismatch: got={got} expected={expected} logits={logits:?}"
        ));
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_q5k_logit_argmax_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    use rnb_backend_vulkan::QuantType;
    use rnb_cpu::quantize::blocks::BlockQ5_K;
    use rnb_cpu::quantize::dequant::dequantize_q5_k;

    const VOCAB: usize = 17;
    const HIDDEN: usize = 256;

    let hidden: Vec<f32> = (0..HIDDEN)
        .map(|i| (((i * 31 + 17) % 131) as f32 - 65.0) * 0.00390625)
        .collect();
    let mut table = Vec::with_capacity(VOCAB * std::mem::size_of::<BlockQ5_K>());
    let mut logits = Vec::with_capacity(VOCAB);

    for row in 0..VOCAB {
        let mut scales = [0u8; 12];
        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        for (index, value) in scales.iter_mut().enumerate() {
            *value = ((index * 23 + row * 29 + 5) & 0xff) as u8;
        }
        for (index, value) in qh.iter_mut().enumerate() {
            *value = ((index * 37 + row * 41 + 3) & 0xff) as u8;
        }
        for (index, value) in qs.iter_mut().enumerate() {
            let low = ((index * 7 + row * 5) & 0x0f) as u8;
            let high = ((index * 13 + row * 11 + 2) & 0x0f) as u8;
            *value = low | (high << 4);
        }
        let block = BlockQ5_K {
            d: half::f16::from_f32(0.0009765625 * (row + 1) as f32),
            dmin: half::f16::from_f32(0.00048828125 * (row + 2) as f32),
            scales,
            qh,
            qs,
        };
        let mut values = [0.0f32; HIDDEN];
        dequantize_q5_k(&block, &mut values);
        logits.push(
            values
                .iter()
                .zip(&hidden)
                .map(|(weight, input)| weight * input)
                .sum::<f32>(),
        );
        append_block_bytes(&mut table, &block);
    }

    let expected = logits
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index as u32)
        .expect("VOCAB is nonzero");
    let got = vk.logit_argmax(&hidden, &table, QuantType::Q5K, VOCAB as u32, HIDDEN as u32)?;
    if got != expected {
        return Err(format!(
            "Q5_K logit argmax mismatch: got={got} expected={expected} logits={logits:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_as_all_suite() {
        let suites = parse_requested_suites(&[]).unwrap();

        assert_eq!(suites, vec![SmokeSuite::All]);
    }

    #[test]
    fn parses_named_suites_in_order() {
        let suites = parse_requested_suites(&["gdn", "attention"]).unwrap();

        assert_eq!(suites, vec![SmokeSuite::Gdn, SmokeSuite::Attention]);
    }

    #[test]
    fn rejects_unknown_suite() {
        let err = parse_requested_suites(&["nope"]).unwrap_err();

        assert!(err.contains("unknown suite"));
    }
}
