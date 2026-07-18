#![cfg(target_arch = "aarch64")]

use std::hint::black_box;
use std::time::Instant;

use half::f16;
use rnb_cpu::quantize::blocks::BlockQ4_K;
use rnb_cpu::quantize::moe_blocks::{GUPairQ4K, GUPairQ4KScaleMin, GUPairQ4KUnpackedScales};
use rnb_cpu::quantize::moe_convert::row_q4k_to_intscale;
use rnb_loader::rnb_moe_reader::{
    RnbMoeView, GATE_UP_QUANT_Q4K_PAIR, GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE,
    GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES,
};

use rnb_cpu::gemm::neon_moe::{
    q4k_gate_block_neon, q4k_gu_block_min_bsum_only, q4k_gu_block_nibble_dot_only_neon,
    q4k_gu_block_unpack_checksum, q4k_up_block_neon, sdot_q4k_gu_block_neon,
    sdot_q4k_gu_block_scale_min_neon, sdot_q4k_gu_block_unpacked_scales_neon,
};
use rnb_cpu::gemm::{quantize_input_q8k, Q8KBlock};

#[derive(Debug, Clone, Copy)]
pub struct Q4KGuBenchConfig {
    pub blocks: usize,
    pub iters: u64,
    pub warmup_iters: u64,
    pub repeats: usize,
}

impl Default for Q4KGuBenchConfig {
    fn default() -> Self {
        Self {
            blocks: 8,
            iters: 300_000,
            warmup_iters: 20_000,
            repeats: 3,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Q4KGuRowBenchConfig {
    pub rows: usize,
    pub blocks_per_row: usize,
    pub selected_rows: usize,
    pub iters: u64,
    pub warmup_iters: u64,
    pub repeats: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Q4KGuSidecarBenchConfig {
    pub layer: usize,
    pub layer_count: usize,
    pub first_expert: usize,
    pub selected_experts: usize,
    pub iters: u64,
    pub warmup_iters: u64,
    pub repeats: usize,
}

#[derive(Debug, Clone)]
pub struct Q4KGuRouteSelection {
    pub layer: usize,
    pub experts: Vec<usize>,
}

impl Default for Q4KGuSidecarBenchConfig {
    fn default() -> Self {
        Self {
            layer: 0,
            layer_count: 1,
            first_expert: 0,
            selected_experts: 8,
            iters: 10,
            warmup_iters: 1,
            repeats: 3,
        }
    }
}

impl Default for Q4KGuRowBenchConfig {
    fn default() -> Self {
        Self {
            rows: 512,
            blocks_per_row: 8,
            selected_rows: 8,
            iters: 200,
            warmup_iters: 10,
            repeats: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Q4KGuBenchResult {
    pub variant: &'static str,
    pub repeat: usize,
    pub elapsed_ns: u128,
    pub ns_per_iter: f64,
    pub ns_per_row: f64,
    pub ns_per_block: f64,
    pub checksum: i64,
}

#[derive(Clone, Copy)]
enum Variant {
    FullGateUp,
    FullUnpackedScales,
    GateOnlyFull,
    NibbleDotOnly,
    MinBsumOnly,
    UnpackOnly,
}

impl Variant {
    const ALL: [Variant; 6] = [
        Variant::FullGateUp,
        Variant::FullUnpackedScales,
        Variant::GateOnlyFull,
        Variant::NibbleDotOnly,
        Variant::MinBsumOnly,
        Variant::UnpackOnly,
    ];

    fn name(self) -> &'static str {
        match self {
            Variant::FullGateUp => "full_gate_up",
            Variant::FullUnpackedScales => "full_unpacked_scales",
            Variant::GateOnlyFull => "gate_only_full",
            Variant::NibbleDotOnly => "nibble_dot_only",
            Variant::MinBsumOnly => "min_bsum_only",
            Variant::UnpackOnly => "unpack_only",
        }
    }
}

#[derive(Clone, Copy)]
enum RowVariant {
    FullGateUp,
    FullUnpackedScales,
    EngineGateUp,
    EngineUnpackedScales,
    GateOnly,
    UpOnly,
}

impl RowVariant {
    const ALL: [RowVariant; 6] = [
        RowVariant::FullGateUp,
        RowVariant::FullUnpackedScales,
        RowVariant::EngineGateUp,
        RowVariant::EngineUnpackedScales,
        RowVariant::GateOnly,
        RowVariant::UpOnly,
    ];

    fn name(self) -> &'static str {
        match self {
            RowVariant::FullGateUp => "row_full_gate_up",
            RowVariant::FullUnpackedScales => "row_full_unpacked_scales",
            RowVariant::EngineGateUp => "row_engine_gate_up",
            RowVariant::EngineUnpackedScales => "row_engine_unpacked_scales",
            RowVariant::GateOnly => "row_gate_only",
            RowVariant::UpOnly => "row_up_only",
        }
    }
}

#[derive(Clone, Copy)]
enum SidecarVariant {
    EngineQ4KPair,
    EngineUnpackedScales,
    EngineScalePlane,
}

impl SidecarVariant {
    fn name(self) -> &'static str {
        match self {
            SidecarVariant::EngineQ4KPair => "sidecar_engine_q4k_pair",
            SidecarVariant::EngineUnpackedScales => "sidecar_engine_unpacked_scales",
            SidecarVariant::EngineScalePlane => "sidecar_engine_scale_plane",
        }
    }
}

#[derive(Clone, Copy)]
struct RowMul {
    gate: f32,
    up: f32,
}

pub fn run_q4k_gu_bench(config: Q4KGuBenchConfig) -> Vec<Q4KGuBenchResult> {
    assert!(config.blocks > 0, "blocks must be > 0");
    assert!(config.iters > 0, "iters must be > 0");
    assert!(config.repeats > 0, "repeats must be > 0");

    let (row, unpacked_row, h_q8k) = build_fixture(config.blocks);
    for variant in Variant::ALL {
        let _ = run_variant(variant, &row, &unpacked_row, &h_q8k, config.warmup_iters);
    }

    let mut out = Vec::with_capacity(config.repeats * Variant::ALL.len());
    for repeat in 0..config.repeats {
        for variant in Variant::ALL {
            let t0 = Instant::now();
            let checksum = run_variant(variant, &row, &unpacked_row, &h_q8k, config.iters);
            let elapsed = t0.elapsed();
            let elapsed_ns = elapsed.as_nanos();
            let denom_iters = config.iters as f64;
            let denom_blocks = (config.iters as f64) * (config.blocks as f64);
            out.push(Q4KGuBenchResult {
                variant: variant.name(),
                repeat,
                elapsed_ns,
                ns_per_iter: elapsed_ns as f64 / denom_iters,
                ns_per_row: elapsed_ns as f64 / denom_iters,
                ns_per_block: elapsed_ns as f64 / denom_blocks,
                checksum,
            });
        }
    }
    out
}

pub fn run_q4k_gu_row_bench(config: Q4KGuRowBenchConfig) -> Vec<Q4KGuBenchResult> {
    assert!(config.rows > 0, "rows must be > 0");
    assert!(config.blocks_per_row > 0, "blocks_per_row must be > 0");
    assert!(config.selected_rows > 0, "selected_rows must be > 0");
    assert!(config.iters > 0, "iters must be > 0");
    assert!(config.repeats > 0, "repeats must be > 0");

    let (rows, unpacked_rows, row_muls, h_q8k) =
        build_row_fixture(config.rows, config.blocks_per_row, config.selected_rows);
    for variant in RowVariant::ALL {
        let _ = run_row_variant(
            variant,
            &rows,
            &unpacked_rows,
            &row_muls,
            &h_q8k,
            config.rows,
            config.blocks_per_row,
            config.selected_rows,
            config.warmup_iters,
        );
    }

    let mut out = Vec::with_capacity(config.repeats * RowVariant::ALL.len());
    for repeat in 0..config.repeats {
        for variant in RowVariant::ALL {
            let t0 = Instant::now();
            let checksum = run_row_variant(
                variant,
                &rows,
                &unpacked_rows,
                &row_muls,
                &h_q8k,
                config.rows,
                config.blocks_per_row,
                config.selected_rows,
                config.iters,
            );
            let elapsed_ns = t0.elapsed().as_nanos();
            let denom_iters = config.iters as f64;
            let row_visits =
                (config.iters as f64) * (config.selected_rows as f64) * (config.rows as f64);
            let block_visits = row_visits * config.blocks_per_row as f64;
            out.push(Q4KGuBenchResult {
                variant: variant.name(),
                repeat,
                elapsed_ns,
                ns_per_iter: elapsed_ns as f64 / denom_iters,
                ns_per_row: elapsed_ns as f64 / row_visits,
                ns_per_block: elapsed_ns as f64 / block_visits,
                checksum,
            });
        }
    }
    out
}

pub fn run_q4k_gu_sidecar_bench(
    file_bytes: &[u8],
    config: Q4KGuSidecarBenchConfig,
    route_trace: Option<&[Q4KGuRouteSelection]>,
) -> Result<Vec<Q4KGuBenchResult>, String> {
    if config.layer_count == 0 {
        return Err("layer_count must be > 0".to_string());
    }
    if config.selected_experts == 0 {
        return Err("selected_experts must be > 0".to_string());
    }
    if config.iters == 0 {
        return Err("iters must be > 0".to_string());
    }
    if config.repeats == 0 {
        return Err("repeats must be > 0".to_string());
    }

    let view = RnbMoeView::from_bytes(file_bytes)?;
    let parsed = view
        .parse_moe_decode()
        .ok_or_else(|| "MOE_DECODE section missing".to_string())??;
    let end_layer = config
        .layer
        .checked_add(config.layer_count)
        .ok_or_else(|| "layer + layer_count overflow".to_string())?;
    if end_layer > parsed.layers.len() {
        return Err(format!(
            "requested layers {}..{} but sidecar has {} layers",
            config.layer,
            end_layer,
            parsed.layers.len()
        ));
    }

    let first_layer = &parsed.layers[config.layer];
    let variant = match first_layer.gate_up_quant {
        GATE_UP_QUANT_Q4K_PAIR => SidecarVariant::EngineQ4KPair,
        GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES => SidecarVariant::EngineUnpackedScales,
        GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE => SidecarVariant::EngineScalePlane,
        other => return Err(format!("unsupported gate_up_quant {other:#x}")),
    };
    let blocks_per_row = (first_layer.n_embd as usize) / 256;

    let h_q8k = quantize_input_q8k(&build_activation(blocks_per_row));
    let _ = run_sidecar_variant(
        variant,
        &parsed.layers,
        &h_q8k,
        config.warmup_iters,
        config,
        route_trace,
    )?;

    let mut out = Vec::with_capacity(config.repeats);
    for repeat in 0..config.repeats {
        let t0 = Instant::now();
        let checksum = run_sidecar_variant(
            variant,
            &parsed.layers,
            &h_q8k,
            config.iters,
            config,
            route_trace,
        )?;
        let elapsed_ns = t0.elapsed().as_nanos();

        let row_visits =
            sidecar_row_visits(&parsed.layers, config, route_trace)? as f64 * config.iters as f64;
        let block_visits =
            sidecar_block_visits(&parsed.layers, config, route_trace)? as f64 * config.iters as f64;
        out.push(Q4KGuBenchResult {
            variant: variant.name(),
            repeat,
            elapsed_ns,
            ns_per_iter: elapsed_ns as f64 / config.iters as f64,
            ns_per_row: elapsed_ns as f64 / row_visits,
            ns_per_block: elapsed_ns as f64 / block_visits,
            checksum,
        });
    }
    Ok(out)
}

#[inline(never)]
fn run_variant(
    variant: Variant,
    row: &[GUPairQ4K],
    unpacked_row: &[GUPairQ4KUnpackedScales],
    h_q8k: &[Q8KBlock],
    iters: u64,
) -> i64 {
    if iters == 0 {
        return 0;
    }

    let mut checksum = 0i64;
    for iter in 0..iters {
        let mut acc = iter as i64;
        for block_idx in 0..row.len() {
            let pair = black_box(&row[block_idx]);
            let unpacked = black_box(&unpacked_row[block_idx]);
            let x = black_box(&h_q8k[block_idx]);
            match variant {
                Variant::FullGateUp => {
                    let (g, u) = unsafe { sdot_q4k_gu_block_neon(pair, x) };
                    acc = acc.wrapping_add(g).wrapping_sub(u.rotate_left(7));
                }
                Variant::FullUnpackedScales => {
                    let (g, u) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(unpacked, x) };
                    acc = acc.wrapping_add(g).wrapping_sub(u.rotate_left(7));
                }
                Variant::GateOnlyFull => {
                    let g = unsafe { q4k_gate_block_neon(pair, x) };
                    acc = acc.wrapping_add(g);
                }
                Variant::NibbleDotOnly => {
                    let (g, u) = unsafe { q4k_gu_block_nibble_dot_only_neon(pair, x) };
                    acc = acc
                        .wrapping_add(g as i64)
                        .wrapping_sub((u as i64).rotate_left(5));
                }
                Variant::MinBsumOnly => {
                    let (g, u) = q4k_gu_block_min_bsum_only(pair, x);
                    acc = acc.wrapping_add(g).wrapping_sub(u.rotate_left(3));
                }
                Variant::UnpackOnly => {
                    acc = acc.wrapping_add(q4k_gu_block_unpack_checksum(pair));
                }
            }
        }
        checksum = checksum.wrapping_add(black_box(acc));
    }
    black_box(checksum)
}

#[inline(never)]
fn run_row_variant(
    variant: RowVariant,
    rows: &[GUPairQ4K],
    unpacked_rows: &[GUPairQ4KUnpackedScales],
    row_muls: &[RowMul],
    h_q8k: &[Q8KBlock],
    row_count: usize,
    blocks_per_row: usize,
    selected_rows: usize,
    iters: u64,
) -> i64 {
    if iters == 0 {
        return 0;
    }

    let mut checksum = 0i64;
    let mut inter = vec![0.0f32; row_count];
    for iter in 0..iters {
        let mut acc = iter as i64;
        for selected_idx in 0..selected_rows {
            let row_shift = ((iter as usize) + selected_idx * 17) % row_count;
            for logical_row in 0..row_count {
                let row_idx = (logical_row + row_shift) % row_count;
                let row_start = (selected_idx * row_count + row_idx) * blocks_per_row;
                match variant {
                    RowVariant::FullGateUp => {
                        for block_idx in 0..blocks_per_row {
                            let pair = black_box(&rows[row_start + block_idx]);
                            let x = black_box(&h_q8k[block_idx]);
                            let (g, u) = unsafe { sdot_q4k_gu_block_neon(pair, x) };
                            acc = acc.wrapping_add(g).wrapping_sub(u.rotate_left(7));
                        }
                    }
                    RowVariant::FullUnpackedScales => {
                        for block_idx in 0..blocks_per_row {
                            let unpacked = black_box(&unpacked_rows[row_start + block_idx]);
                            let x = black_box(&h_q8k[block_idx]);
                            let (g, u) =
                                unsafe { sdot_q4k_gu_block_unpacked_scales_neon(unpacked, x) };
                            acc = acc.wrapping_add(g).wrapping_sub(u.rotate_left(7));
                        }
                    }
                    RowVariant::EngineGateUp => {
                        let row_mul = row_muls[selected_idx * row_count + row_idx];
                        let mut g_acc = 0.0f32;
                        let mut u_acc = 0.0f32;
                        for block_idx in 0..blocks_per_row {
                            let pair = black_box(&rows[row_start + block_idx]);
                            let x = black_box(&h_q8k[block_idx]);
                            let (g, u) = unsafe { sdot_q4k_gu_block_neon(pair, x) };
                            let d = x.d;
                            g_acc += (g as f32) * d;
                            u_acc += (u as f32) * d;
                        }
                        let value = silu_f32(g_acc * row_mul.gate) * (u_acc * row_mul.up);
                        inter[row_idx] = black_box(value);
                        acc = acc.wrapping_add(value.to_bits() as i64);
                    }
                    RowVariant::EngineUnpackedScales => {
                        let row_mul = row_muls[selected_idx * row_count + row_idx];
                        let mut g_acc = 0.0f32;
                        let mut u_acc = 0.0f32;
                        for block_idx in 0..blocks_per_row {
                            let unpacked = black_box(&unpacked_rows[row_start + block_idx]);
                            let x = black_box(&h_q8k[block_idx]);
                            let (g, u) =
                                unsafe { sdot_q4k_gu_block_unpacked_scales_neon(unpacked, x) };
                            let d = x.d;
                            g_acc += (g as f32) * d;
                            u_acc += (u as f32) * d;
                        }
                        let value = silu_f32(g_acc * row_mul.gate) * (u_acc * row_mul.up);
                        inter[row_idx] = black_box(value);
                        acc = acc.wrapping_add(value.to_bits() as i64);
                    }
                    RowVariant::GateOnly => {
                        for block_idx in 0..blocks_per_row {
                            let pair = black_box(&rows[row_start + block_idx]);
                            let x = black_box(&h_q8k[block_idx]);
                            let g = unsafe { q4k_gate_block_neon(pair, x) };
                            acc = acc.wrapping_add(g);
                        }
                    }
                    RowVariant::UpOnly => {
                        for block_idx in 0..blocks_per_row {
                            let pair = black_box(&rows[row_start + block_idx]);
                            let x = black_box(&h_q8k[block_idx]);
                            let u = unsafe { q4k_up_block_neon(pair, x) };
                            acc = acc.wrapping_sub(u.rotate_left(7));
                        }
                    }
                }
            }
        }
        checksum = checksum.wrapping_add(black_box(acc));
    }
    black_box(checksum)
}

#[inline(never)]
fn run_sidecar_variant(
    variant: SidecarVariant,
    layers: &[rnb_loader::rnb_moe_reader::MoeDecodeLayer<'_>],
    h_q8k: &[Q8KBlock],
    iters: u64,
    config: Q4KGuSidecarBenchConfig,
    route_trace: Option<&[Q4KGuRouteSelection]>,
) -> Result<i64, String> {
    if iters == 0 {
        return Ok(0);
    }

    let mut checksum = 0i64;
    let end_layer = config.layer + config.layer_count;
    for iter in 0..iters {
        let mut acc = iter as i64;
        if let Some(trace) = route_trace {
            for selection in trace {
                if selection.layer < config.layer || selection.layer >= end_layer {
                    continue;
                }
                let layer = &layers[selection.layer];
                for &expert_idx in &selection.experts {
                    run_sidecar_expert_rows(
                        variant,
                        selection.layer,
                        expert_idx,
                        layer,
                        h_q8k,
                        &mut acc,
                    )?;
                }
            }
        } else {
            for layer_idx in config.layer..end_layer {
                let layer = &layers[layer_idx];
                if layer.experts.is_empty() {
                    return Err(format!("layer {layer_idx}: no experts"));
                }
                for selected_idx in 0..config.selected_experts {
                    let expert_idx =
                        (config.first_expert + selected_idx + iter as usize) % layer.experts.len();
                    run_sidecar_expert_rows(
                        variant, layer_idx, expert_idx, layer, h_q8k, &mut acc,
                    )?;
                }
            }
        }
        checksum = checksum.wrapping_add(black_box(acc));
    }
    Ok(black_box(checksum))
}

fn run_sidecar_expert_rows(
    variant: SidecarVariant,
    layer_idx: usize,
    expert_idx: usize,
    layer: &rnb_loader::rnb_moe_reader::MoeDecodeLayer<'_>,
    h_q8k: &[Q8KBlock],
    acc: &mut i64,
) -> Result<(), String> {
    let blocks_per_row = layer.n_embd as usize / 256;
    if blocks_per_row != h_q8k.len() {
        return Err(format!(
            "layer {layer_idx}: n_embd blocks {} != activation blocks {}",
            blocks_per_row,
            h_q8k.len()
        ));
    }
    let Some(expert) = layer.experts.get(expert_idx) else {
        return Err(format!(
            "layer {layer_idx}: expert {expert_idx} out of range {}",
            layer.experts.len()
        ));
    };
    for (row_idx, row) in expert.gate_up_rows.iter().enumerate() {
        let mut g_acc = 0.0f32;
        let mut u_acc = 0.0f32;
        match variant {
            SidecarVariant::EngineQ4KPair => {
                let expected_len = blocks_per_row * std::mem::size_of::<GUPairQ4K>();
                if row.blocks_bytes.len() != expected_len {
                    return Err(format!(
                        "layer {layer_idx} expert {expert_idx} row {row_idx}: legacy row bytes {} != {expected_len}",
                        row.blocks_bytes.len()
                    ));
                }
                let row_ptr = row.blocks_bytes.as_ptr() as *const GUPairQ4K;
                for block_idx in 0..blocks_per_row {
                    let pair = unsafe { &*row_ptr.add(block_idx) };
                    let x = black_box(&h_q8k[block_idx]);
                    let (g, u) = unsafe { sdot_q4k_gu_block_neon(pair, x) };
                    let d = x.d;
                    g_acc += (g as f32) * d;
                    u_acc += (u as f32) * d;
                }
            }
            SidecarVariant::EngineUnpackedScales => {
                let expected_len = blocks_per_row * std::mem::size_of::<GUPairQ4KUnpackedScales>();
                if row.blocks_bytes.len() != expected_len {
                    return Err(format!(
                        "layer {layer_idx} expert {expert_idx} row {row_idx}: unpacked row bytes {} != {expected_len}",
                        row.blocks_bytes.len()
                    ));
                }
                let row_ptr = row.blocks_bytes.as_ptr() as *const GUPairQ4KUnpackedScales;
                for block_idx in 0..blocks_per_row {
                    let pair = unsafe { &*row_ptr.add(block_idx) };
                    let x = black_box(&h_q8k[block_idx]);
                    let (g, u) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(pair, x) };
                    let d = x.d;
                    g_acc += (g as f32) * d;
                    u_acc += (u as f32) * d;
                }
            }
            SidecarVariant::EngineScalePlane => {
                let expected_len = blocks_per_row * std::mem::size_of::<GUPairQ4K>();
                if row.blocks_bytes.len() != expected_len {
                    return Err(format!(
                        "layer {layer_idx} expert {expert_idx} row {row_idx}: scale-plane row bytes {} != {expected_len}",
                        row.blocks_bytes.len()
                    ));
                }
                let scale_bytes = row.scale_bytes.ok_or_else(|| {
                    format!(
                        "layer {layer_idx} expert {expert_idx} row {row_idx}: scale-plane row missing side bytes"
                    )
                })?;
                let expected_scale_len = blocks_per_row * std::mem::size_of::<GUPairQ4KScaleMin>();
                if scale_bytes.len() != expected_scale_len {
                    return Err(format!(
                        "layer {layer_idx} expert {expert_idx} row {row_idx}: scale-plane side bytes {} != {expected_scale_len}",
                        scale_bytes.len()
                    ));
                }
                let row_ptr = row.blocks_bytes.as_ptr() as *const GUPairQ4K;
                let scale_ptr = scale_bytes.as_ptr() as *const GUPairQ4KScaleMin;
                for block_idx in 0..blocks_per_row {
                    let pair = unsafe { &*row_ptr.add(block_idx) };
                    let scale = unsafe { &*scale_ptr.add(block_idx) };
                    let x = black_box(&h_q8k[block_idx]);
                    let (g, u) = unsafe { sdot_q4k_gu_block_scale_min_neon(pair, scale, x) };
                    let d = x.d;
                    g_acc += (g as f32) * d;
                    u_acc += (u as f32) * d;
                }
            }
        }
        let value = silu_f32(g_acc * row.gate_mul) * (u_acc * row.up_mul);
        *acc = acc.wrapping_add(value.to_bits() as i64);
    }
    Ok(())
}

fn sidecar_row_visits(
    layers: &[rnb_loader::rnb_moe_reader::MoeDecodeLayer<'_>],
    config: Q4KGuSidecarBenchConfig,
    route_trace: Option<&[Q4KGuRouteSelection]>,
) -> Result<usize, String> {
    let end_layer = config.layer + config.layer_count;
    let mut rows = 0usize;
    if let Some(trace) = route_trace {
        for selection in trace {
            if selection.layer < config.layer || selection.layer >= end_layer {
                continue;
            }
            let layer = layers
                .get(selection.layer)
                .ok_or_else(|| format!("route layer {} out of range", selection.layer))?;
            for &expert_idx in &selection.experts {
                if expert_idx >= layer.experts.len() {
                    return Err(format!(
                        "layer {}: expert {expert_idx} out of range {}",
                        selection.layer,
                        layer.experts.len()
                    ));
                }
                rows = rows
                    .checked_add(layer.d_ff as usize)
                    .ok_or_else(|| "row visit count overflow".to_string())?;
            }
        }
    } else {
        for layer_idx in config.layer..end_layer {
            let layer = &layers[layer_idx];
            if layer.experts.is_empty() {
                return Err(format!("layer {layer_idx}: no experts"));
            }
            rows = rows
                .checked_add((layer.d_ff as usize) * config.selected_experts)
                .ok_or_else(|| "row visit count overflow".to_string())?;
        }
    }
    if rows == 0 {
        return Err("route trace has no selections in requested layer range".to_string());
    }
    Ok(rows)
}

fn sidecar_block_visits(
    layers: &[rnb_loader::rnb_moe_reader::MoeDecodeLayer<'_>],
    config: Q4KGuSidecarBenchConfig,
    route_trace: Option<&[Q4KGuRouteSelection]>,
) -> Result<usize, String> {
    let end_layer = config.layer + config.layer_count;
    let mut blocks = 0usize;
    if let Some(trace) = route_trace {
        for selection in trace {
            if selection.layer < config.layer || selection.layer >= end_layer {
                continue;
            }
            let layer = layers
                .get(selection.layer)
                .ok_or_else(|| format!("route layer {} out of range", selection.layer))?;
            let blocks_per_row = layer.n_embd as usize / 256;
            for &expert_idx in &selection.experts {
                if expert_idx >= layer.experts.len() {
                    return Err(format!(
                        "layer {}: expert {expert_idx} out of range {}",
                        selection.layer,
                        layer.experts.len()
                    ));
                }
                blocks = blocks
                    .checked_add((layer.d_ff as usize) * blocks_per_row)
                    .ok_or_else(|| "block visit count overflow".to_string())?;
            }
        }
    } else {
        for layer_idx in config.layer..end_layer {
            let layer = &layers[layer_idx];
            let blocks_per_row = layer.n_embd as usize / 256;
            blocks = blocks
                .checked_add((layer.d_ff as usize) * config.selected_experts * blocks_per_row)
                .ok_or_else(|| "block visit count overflow".to_string())?;
        }
    }
    if blocks == 0 {
        return Err("route trace has no selections in requested layer range".to_string());
    }
    Ok(blocks)
}

#[inline]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn build_fixture(n_blocks: usize) -> (Vec<GUPairQ4K>, Vec<GUPairQ4KUnpackedScales>, Vec<Q8KBlock>) {
    let gate_blocks: Vec<BlockQ4_K> = (0..n_blocks).map(|i| synth_q4k_block(i, 0)).collect();
    let up_blocks: Vec<BlockQ4_K> = (0..n_blocks).map(|i| synth_q4k_block(i, 53)).collect();
    let (gate_is, _gate_mul) = row_q4k_to_intscale(&gate_blocks);
    let (up_is, _up_mul) = row_q4k_to_intscale(&up_blocks);
    let row: Vec<GUPairQ4K> = gate_is
        .into_iter()
        .zip(up_is)
        .map(|(gate, up)| GUPairQ4K { gate, up })
        .collect();
    let unpacked_row = row
        .iter()
        .copied()
        .map(GUPairQ4KUnpackedScales::from_pair)
        .collect();
    let activation = build_activation(n_blocks);
    let h_q8k = quantize_input_q8k(&activation);
    (row, unpacked_row, h_q8k)
}

fn build_row_fixture(
    rows: usize,
    blocks_per_row: usize,
    selected_rows: usize,
) -> (
    Vec<GUPairQ4K>,
    Vec<GUPairQ4KUnpackedScales>,
    Vec<RowMul>,
    Vec<Q8KBlock>,
) {
    let mut row_units = Vec::with_capacity(selected_rows * rows * blocks_per_row);
    let mut row_muls = Vec::with_capacity(selected_rows * rows);
    for selected_idx in 0..selected_rows {
        for row_idx in 0..rows {
            let base = (selected_idx * rows + row_idx) * blocks_per_row;
            let gate_blocks: Vec<BlockQ4_K> = (0..blocks_per_row)
                .map(|block_idx| synth_q4k_block(base + block_idx, 0))
                .collect();
            let up_blocks: Vec<BlockQ4_K> = (0..blocks_per_row)
                .map(|block_idx| synth_q4k_block(base + block_idx, 53))
                .collect();
            let (gate_is, gate_mul) = row_q4k_to_intscale(&gate_blocks);
            let (up_is, up_mul) = row_q4k_to_intscale(&up_blocks);
            row_muls.push(RowMul {
                gate: gate_mul,
                up: up_mul,
            });
            row_units.extend(
                gate_is
                    .into_iter()
                    .zip(up_is)
                    .map(|(gate, up)| GUPairQ4K { gate, up }),
            );
        }
    }
    let unpacked_rows = row_units
        .iter()
        .copied()
        .map(GUPairQ4KUnpackedScales::from_pair)
        .collect();
    let activation = build_activation(blocks_per_row);
    let h_q8k = quantize_input_q8k(&activation);
    (row_units, unpacked_rows, row_muls, h_q8k)
}

fn synth_q4k_block(idx: usize, offset: u8) -> BlockQ4_K {
    let mut qs = [0u8; 128];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = ((i as u8).wrapping_add(offset))
            .wrapping_add((idx as u8).wrapping_mul(11))
            .wrapping_mul(17);
    }
    let sc = [
        1 + (idx as u8 & 3),
        7,
        12,
        19,
        27,
        34,
        45,
        61 - (idx as u8 & 3),
    ];
    let m = [
        2,
        9 + (idx as u8 & 3),
        14,
        23,
        31,
        40,
        52,
        62 - (idx as u8 & 3),
    ];
    BlockQ4_K {
        d: f16::from_f32(0.05 * (1.0 + 0.1 * idx as f32)),
        dmin: f16::from_f32(0.02 * (1.0 + 0.05 * idx as f32)),
        scales: pack_q4k_sub_scales(&sc, &m),
        qs,
    }
}

fn pack_q4k_sub_scales(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = (sc[j] & 0x3F) | (((sc[j + 4] >> 4) & 0x03) << 6);
        scales[j + 4] = (m[j] & 0x3F) | (((m[j + 4] >> 4) & 0x03) << 6);
        scales[j + 8] = (sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4);
    }
    scales
}

fn build_activation(n_blocks: usize) -> Vec<f32> {
    let n = n_blocks * 256;
    (0..n)
        .map(|i| {
            let t = i as f32 * 0.0137;
            t.sin() + 0.4 * (t * 1.7).cos() - 0.3 * (t * 0.3).sin()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_bench_config_defaults_to_qwen35_shape() {
        let config = Q4KGuRowBenchConfig::default();

        assert_eq!(config.rows, 512);
        assert_eq!(config.blocks_per_row, 8);
        assert_eq!(config.selected_rows, 8);
        assert_eq!(config.iters, 200);
        assert_eq!(config.warmup_iters, 10);
        assert_eq!(config.repeats, 3);
    }

    #[test]
    fn row_bench_reports_per_row_and_per_block_metrics() {
        let config = Q4KGuRowBenchConfig {
            rows: 4,
            blocks_per_row: 2,
            selected_rows: 3,
            iters: 2,
            warmup_iters: 1,
            repeats: 1,
        };

        let results = run_q4k_gu_row_bench(config);
        let variants: Vec<&str> = results.iter().map(|r| r.variant).collect();

        assert_eq!(
            variants,
            vec![
                "row_full_gate_up",
                "row_full_unpacked_scales",
                "row_engine_gate_up",
                "row_engine_unpacked_scales",
                "row_gate_only",
                "row_up_only",
            ]
        );
        for result in results {
            assert_eq!(result.repeat, 0);
            assert!(result.elapsed_ns > 0);
            assert!(result.ns_per_iter > 0.0);
            assert!(result.ns_per_row > 0.0);
            assert!(result.ns_per_block > 0.0);
            assert_ne!(result.checksum, 0);
        }
    }

    #[test]
    fn sidecar_bench_config_defaults_to_one_layer_active8() {
        let config = Q4KGuSidecarBenchConfig::default();

        assert_eq!(config.layer, 0);
        assert_eq!(config.layer_count, 1);
        assert_eq!(config.first_expert, 0);
        assert_eq!(config.selected_experts, 8);
        assert_eq!(config.iters, 10);
        assert_eq!(config.warmup_iters, 1);
        assert_eq!(config.repeats, 3);
    }
}
