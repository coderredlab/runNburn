use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};

use crate::compute::{self, MetalContext};
use crate::{ffn_chain, tests_fixture};

// Collected from Qwen3.6-35B-A3B-Q4_K_M, standard large prompt truncated to
// 128 prefill tokens, layer 0, on 2026-07-16. Each row is layer + top-8 experts.
const QWEN36_PREFILL128_LAYER0_ROUTES: &str = include_str!("qwen36_prefill128_layer0_routes.csv");
const HIDDEN_DIM: usize = 2048;
const FFN_DIM: usize = 3648;
const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_BLOCK_ELEMS: usize = 256;
const WARMUPS: usize = 1;
const SAMPLES: usize = 4;
const MAD_SCALE: f64 = 2.0 * 1.4826;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

struct RouteFixture {
    name: &'static str,
    n_tokens: usize,
    top_k: usize,
    n_expert: usize,
    selected: Vec<u32>,
}

struct ExpertBuffers {
    expert: usize,
    count: usize,
    token_offset: usize,
    input_f16: Buffer,
    output: Buffer,
    m: Buffer,
}

struct Samples {
    gather: Vec<f64>,
    existing_gate: Vec<f64>,
    existing_total: Vec<f64>,
    map: Vec<f64>,
    candidate_gate: Vec<f64>,
    candidate_total: Vec<f64>,
}

impl Samples {
    fn new() -> Self {
        Self {
            gather: Vec::with_capacity(SAMPLES),
            existing_gate: Vec::with_capacity(SAMPLES),
            existing_total: Vec::with_capacity(SAMPLES),
            map: Vec::with_capacity(SAMPLES),
            candidate_gate: Vec::with_capacity(SAMPLES),
            candidate_total: Vec::with_capacity(SAMPLES),
        }
    }
}

fn dense_top2_fixture() -> RouteFixture {
    let n_tokens = 128;
    let top_k = 2;
    let n_expert = 8;
    let mut selected = Vec::with_capacity(n_tokens * top_k);
    for token in 0..n_tokens {
        selected.push((token % n_expert) as u32);
        selected.push(((token * 3 + 1) % n_expert) as u32);
    }
    RouteFixture {
        name: "dense-top2-of-8",
        n_tokens,
        top_k,
        n_expert,
        selected,
    }
}

fn production_sparse_fixture() -> RouteFixture {
    let n_tokens = 128;
    let top_k = 8;
    let n_expert = 256;
    let mut selected = Vec::with_capacity(n_tokens * top_k);
    for (token, line) in QWEN36_PREFILL128_LAYER0_ROUTES.lines().enumerate() {
        let mut fields = line
            .split(',')
            .map(|value| value.parse::<u32>().expect("route fixture integer"));
        assert_eq!(
            fields.next(),
            Some(0),
            "route fixture layer at token {token}"
        );
        let mut seen = [false; 256];
        for rank in 0..top_k {
            let expert = fields.next().expect("route fixture top-k");
            assert!((expert as usize) < n_expert);
            assert!(
                !std::mem::replace(&mut seen[expert as usize], true),
                "duplicate expert {expert} at token {token} rank {rank}"
            );
            selected.push(expert);
        }
        assert_eq!(fields.next(), None, "extra route field at token {token}");
    }
    assert_eq!(selected.len(), n_tokens * top_k);
    RouteFixture {
        name: "qwen36-layer0-top8-of-256",
        n_tokens,
        top_k,
        n_expert,
        selected,
    }
}

fn shared_i32(ctx: &MetalContext, values: &[i32]) -> Buffer {
    unsafe {
        let ptr = NonNull::new(values.as_ptr() as *mut std::ffi::c_void).expect("i32 ptr");
        ctx.device
            .newBufferWithBytes_length_options(
                ptr,
                std::mem::size_of_val(values),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("shared i32 buffer")
    }
}

fn empty_buffer(ctx: &MetalContext, bytes: usize) -> Buffer {
    ctx.device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("shared empty buffer")
}

fn repeated_q4k_buffer(ctx: &MetalContext, block: &[u8], bytes: usize) -> Buffer {
    assert_eq!(block.len(), Q4K_BLOCK_BYTES);
    assert_eq!(bytes % block.len(), 0);
    let buffer = empty_buffer(ctx, bytes);
    unsafe {
        let dst = std::slice::from_raw_parts_mut(buffer.contents().as_ptr() as *mut u8, bytes);
        dst[..block.len()].copy_from_slice(block);
        let mut filled = block.len();
        while filled < bytes {
            let copied = filled.min(bytes - filled);
            std::ptr::copy_nonoverlapping(dst.as_ptr(), dst.as_mut_ptr().add(filled), copied);
            filled += copied;
        }
    }
    buffer
}

fn read_u32(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> &[u32] {
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u32, len) }
}
fn read_i32(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> &[i32] {
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const i32, len) }
}

fn read_f32(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, len) }
}

fn set_u32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: usize, index: usize) {
    let value = u32::try_from(value).expect("u32 Metal argument");
    unsafe {
        let ptr = NonNull::new(&value as *const u32 as *mut std::ffi::c_void)
            .expect("u32 Metal argument ptr");
        enc.setBytes_length_atIndex(ptr, std::mem::size_of::<u32>(), index);
    }
}

fn gpu_ms(
    ctx: &MetalContext,
    encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> f64 {
    let command = ctx.queue.commandBuffer().expect("Metal command buffer");
    let encoder = command
        .computeCommandEncoder()
        .expect("Metal compute command encoder");
    encode(&encoder);
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    ((command.GPUEndTime() - command.GPUStartTime()) * 1000.0).max(0.0)
}

fn encode_map(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    selected: &ProtocolObject<dyn MTLBuffer>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    fixture: &RouteFixture,
) {
    ffn_chain::qwen_moe_llama_id_map0_encode(
        ctx,
        enc,
        selected,
        tpe,
        ids,
        fixture.n_tokens,
        fixture.top_k,
        fixture.n_expert,
    )
    .expect("map0 encode");
}

#[allow(clippy::too_many_arguments)]
fn encode_candidate(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    fixture: &RouteFixture,
    expert_stride: usize,
) {
    ffn_chain::qwen_moe_llama_mul_mm_id_encode(
        ctx,
        enc,
        ffn_chain::QwenMoeLlamaIdQuant::Q4K,
        ffn_chain::QwenMoeLlamaIdInput::F32,
        weight,
        0,
        input,
        tpe,
        ids,
        output,
        ffn_chain::QwenMoeLlamaIdMatmulShape {
            input_dim: HIDDEN_DIM,
            output_dim: FFN_DIM,
            n_tokens: fixture.n_tokens,
            n_expert: fixture.n_expert,
            n_expert_used: fixture.top_k,
            expert_weight_bytes: expert_stride,
        },
    )
    .expect("production Q4_K/F32 ID encode");
}

fn encode_gather(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &ProtocolObject<dyn MTLBuffer>,
    sorted_tokens: &ProtocolObject<dyn MTLBuffer>,
    experts: &[ExpertBuffers],
) {
    let pipeline = &ctx.qwen_moe_prefill_gather_f16_pipeline;
    let threads = pipeline.threadExecutionWidth().max(1);
    for expert in experts {
        let total = expert.count * HIDDEN_DIM;
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&expert.input_f16), 0, 1);
            enc.setBuffer_offset_atIndex(Some(sorted_tokens), 0, 2);
        }
        set_u32(enc, HIDDEN_DIM, 3);
        set_u32(enc, expert.token_offset, 4);
        set_u32(enc, total, 5);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: total.div_ceil(threads),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }
}

fn encode_existing_gate(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    n: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    experts: &[ExpertBuffers],
    expert_stride: usize,
) {
    for expert in experts {
        compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            enc,
            weight,
            u32::try_from(expert.expert * expert_stride).expect("weight byte offset"),
            &expert.input_f16,
            &expert.output,
            n,
            k,
            &expert.m,
            FFN_DIM,
            expert.count,
        );
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

fn mad(values: &[f64]) -> f64 {
    let center = median(values);
    let deviations = values
        .iter()
        .map(|value| (value - center).abs())
        .collect::<Vec<_>>();
    median(&deviations)
}

fn variance_margin(a: &[f64], b: &[f64]) -> f64 {
    MAD_SCALE * (mad(a) + mad(b))
}

fn print_samples(name: &str, samples: &Samples) {
    let existing_gate = median(&samples.existing_gate);
    let candidate_gate = median(&samples.candidate_gate);
    let gate_margin = variance_margin(&samples.existing_gate, &samples.candidate_gate);
    let gate_delta = (candidate_gate / existing_gate - 1.0) * 100.0;
    let gate_green = candidate_gate + gate_margin < existing_gate;
    let map = median(&samples.map);
    let amortized_gate = candidate_gate + map / 3.0;
    let amortized_gate_green = amortized_gate + gate_margin < existing_gate;

    let existing_total = median(&samples.existing_total);
    let candidate_total = median(&samples.candidate_total);
    let total_margin = variance_margin(&samples.existing_total, &samples.candidate_total);
    let total_delta = (candidate_total / existing_total - 1.0) * 100.0;
    let total_green = candidate_total + total_margin < existing_total;

    eprintln!(
        "[qwen-llama-id-microbench:{name}] raw_gather_ms={:?}",
        samples.gather
    );
    eprintln!(
        "[qwen-llama-id-microbench:{name}] raw_existing_gate_ms={:?}",
        samples.existing_gate
    );
    eprintln!(
        "[qwen-llama-id-microbench:{name}] raw_candidate_gate_ms={:?}",
        samples.candidate_gate
    );
    eprintln!(
        "[qwen-llama-id-microbench:{name}] raw_map_ms={:?}",
        samples.map
    );
    eprintln!(
        "[qwen-llama-id-microbench:{name}] raw_existing_gather_plus_gate_ms={:?}",
        samples.existing_total
    );
    eprintln!(
        "[qwen-llama-id-microbench:{name}] raw_map_plus_candidate_gate_ms={:?}",
        samples.candidate_total
    );
    eprintln!(
        "[qwen-llama-id-microbench:{name}] median_gather_ms={:.6} median_existing_gate_ms={:.6} median_map_ms={:.6} median_candidate_gate_ms={:.6} map_amortized_gate_ms={:.6}",
        median(&samples.gather),
        existing_gate,
        map,
        candidate_gate,
        candidate_gate + median(&samples.map) / 3.0,
    );
    eprintln!(
        "[qwen-llama-id-microbench:{name}] gate_mad_existing={:.6} gate_mad_candidate={:.6} variance_margin={:.6} delta_pct={:+.3} verdict={}",
        mad(&samples.existing_gate),
        mad(&samples.candidate_gate),
        gate_margin,
        gate_delta,
        if gate_green { "GREEN" } else { "NO-GO" },
    );
    eprintln!(
        "[qwen-llama-id-microbench:{name}] median_existing_total_ms={:.6} median_candidate_total_ms={:.6} total_variance_margin={:.6} total_delta_pct={:+.3} total_verdict={}",
        existing_total,
        candidate_total,
        total_margin,
        total_delta,
        if total_green { "GREEN" } else { "NO-GO" },
    );
    assert!(
        gate_green,
        "{name} pure gate NO-GO: candidate_gate={candidate_gate:.6} + variance_margin={gate_margin:.6} >= existing_gate={existing_gate:.6}"
    );
    assert!(
        amortized_gate_green,
        "{name} amortized gate NO-GO: candidate_gate={candidate_gate:.6} + map/3={:.6} + variance_margin={gate_margin:.6} >= existing_gate={existing_gate:.6}",
        map / 3.0,
    );
    assert!(
        total_green,
        "{name} total NO-GO: candidate_total={candidate_total:.6} + variance_margin={total_margin:.6} >= existing_total={existing_total:.6}"
    );
}

fn run_fixture(ctx: &MetalContext, fixture: RouteFixture) {
    let slots = fixture.n_tokens * fixture.top_k;
    let (expert_counts, route_slot_ids) = ffn_chain::qwen_moe_llama_map0_cpu(
        &fixture.selected,
        fixture.n_tokens,
        fixture.top_k,
        fixture.n_expert,
    )
    .expect("CPU map0 oracle");
    let max_count = expert_counts.iter().copied().max().unwrap_or(0);
    let nonempty = expert_counts.iter().filter(|&&count| count != 0).count();
    let min_nonempty = expert_counts
        .iter()
        .copied()
        .filter(|&count| count != 0)
        .min()
        .unwrap_or(0);
    let mut histogram = std::collections::BTreeMap::new();
    for &count in &expert_counts {
        *histogram.entry(count).or_insert(0usize) += 1;
    }
    let empty = fixture.n_expert - nonempty;
    eprintln!(
        "[qwen-llama-id-microbench:{}] tokens={} experts={} top_k={} slots={} K={} N={} nonempty={} empty={} empty_ratio_pct={:.3} min_nonempty={} max_count={} count_histogram={:?}",
        fixture.name,
        fixture.n_tokens,
        fixture.n_expert,
        fixture.top_k,
        slots,
        HIDDEN_DIM,
        FFN_DIM,
        nonempty,
        empty,
        empty as f64 / fixture.n_expert as f64 * 100.0,
        min_nonempty,
        max_count,
        histogram,
    );

    let mut sorted_tokens = Vec::with_capacity(slots);
    let mut token_offsets = vec![0usize; fixture.n_expert];
    for expert in 0..fixture.n_expert {
        token_offsets[expert] = sorted_tokens.len();
        for local in 0..expert_counts[expert] as usize {
            let slot = route_slot_ids[expert * fixture.n_tokens + local] as usize;
            sorted_tokens.push((slot / fixture.top_k) as u32);
        }
    }
    assert_eq!(sorted_tokens.len(), slots);

    let expert_stride = FFN_DIM * (HIDDEN_DIM / Q4K_BLOCK_ELEMS) * Q4K_BLOCK_BYTES;
    let weight_bytes = fixture.n_expert * expert_stride;
    let q4k_block = tests_fixture::q4k_block_fixed();
    let weight = repeated_q4k_buffer(ctx, &q4k_block, weight_bytes);
    let input_values = (0..fixture.n_tokens * HIDDEN_DIM)
        .map(|index| 0.01 + (index % 29) as f32 * 0.0005)
        .collect::<Vec<_>>();
    let input = ffn_chain::shared_f32_buf(ctx, &input_values);
    let selected = ffn_chain::shared_u32_buf(ctx, &fixture.selected);
    let sorted_tokens_buf = ffn_chain::shared_u32_buf(ctx, &sorted_tokens);
    let cpu_tpe = ffn_chain::shared_u32_buf(ctx, &expert_counts);
    let cpu_ids = shared_i32(ctx, &route_slot_ids);
    let gpu_tpe = empty_buffer(ctx, fixture.n_expert * std::mem::size_of::<u32>());
    let gpu_ids = empty_buffer(
        ctx,
        fixture.n_expert * fixture.n_tokens * std::mem::size_of::<u32>(),
    );
    let candidate_output = empty_buffer(ctx, slots * FFN_DIM * std::mem::size_of::<f32>());
    let n = ffn_chain::u32_buf(ctx, FFN_DIM as u32);
    let k = ffn_chain::u32_buf(ctx, HIDDEN_DIM as u32);

    let mut experts = Vec::with_capacity(nonempty);
    for expert in 0..fixture.n_expert {
        let count = expert_counts[expert] as usize;
        if count == 0 {
            continue;
        }
        experts.push(ExpertBuffers {
            expert,
            count,
            token_offset: token_offsets[expert],
            input_f16: empty_buffer(ctx, count * HIDDEN_DIM * std::mem::size_of::<u16>()),
            output: empty_buffer(ctx, count * FFN_DIM * std::mem::size_of::<f32>()),
            m: ffn_chain::u32_buf(ctx, count as u32),
        });
    }

    gpu_ms(ctx, |enc| {
        encode_map(ctx, enc, &selected, &gpu_tpe, &gpu_ids, &fixture);
    });
    assert_eq!(read_u32(&gpu_tpe, fixture.n_expert), expert_counts);
    let gpu_ids_values = read_i32(&gpu_ids, fixture.n_expert * fixture.n_tokens);
    for expert in 0..fixture.n_expert {
        let count = expert_counts[expert] as usize;
        let start = expert * fixture.n_tokens;
        assert_eq!(
            &gpu_ids_values[start..start + count],
            &route_slot_ids[start..start + count],
            "GPU map0 mismatch for expert {expert}"
        );
    }

    gpu_ms(ctx, |enc| {
        encode_gather(ctx, enc, &input, &sorted_tokens_buf, &experts);
        encode_existing_gate(ctx, enc, &weight, &n, &k, &experts, expert_stride);
    });
    gpu_ms(ctx, |enc| {
        encode_candidate(
            ctx,
            enc,
            &weight,
            &input,
            &cpu_tpe,
            &cpu_ids,
            &candidate_output,
            &fixture,
            expert_stride,
        );
    });

    let candidate_values = read_f32(&candidate_output, slots * FFN_DIM);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_magnitude = 0.0f32;
    let mut first_pair = None;
    let mut worst = (0usize, 0usize, 0usize, 0.0f32, 0.0f32);
    for expert in &experts {
        let baseline_values = read_f32(&expert.output, expert.count * FFN_DIM);
        for local in 0..expert.count {
            let slot = route_slot_ids[expert.expert * fixture.n_tokens + local] as usize;
            for row in 0..FFN_DIM {
                let baseline = baseline_values[local * FFN_DIM + row];
                let candidate = candidate_values[slot * FFN_DIM + row];
                first_pair.get_or_insert((baseline, candidate));
                max_magnitude = max_magnitude.max(baseline.abs()).max(candidate.abs());
                let abs = (candidate - baseline).abs();
                max_abs = max_abs.max(abs);
                let rel = abs / baseline.abs().max(1e-6);
                if rel > max_rel {
                    max_rel = rel;
                    worst = (expert.expert, local, row, baseline, candidate);
                }
            }
        }
    }
    eprintln!(
        "[qwen-llama-id-microbench:{}] parity max_rel={:.6e} max_abs={:.6e} max_magnitude={:.6e} first={:?} worst={:?}",
        fixture.name, max_rel, max_abs, max_magnitude, first_pair, worst
    );
    assert!(
        max_rel < 1e-3,
        "{} candidate mismatch: max_rel={max_rel} max_abs={max_abs} worst={worst:?}",
        fixture.name
    );

    for _ in 0..WARMUPS {
        gpu_ms(ctx, |enc| {
            encode_gather(ctx, enc, &input, &sorted_tokens_buf, &experts)
        });
        gpu_ms(ctx, |enc| {
            encode_existing_gate(ctx, enc, &weight, &n, &k, &experts, expert_stride)
        });
        gpu_ms(ctx, |enc| {
            encode_gather(ctx, enc, &input, &sorted_tokens_buf, &experts);
            encode_existing_gate(ctx, enc, &weight, &n, &k, &experts, expert_stride);
        });
        gpu_ms(ctx, |enc| {
            encode_map(ctx, enc, &selected, &gpu_tpe, &gpu_ids, &fixture)
        });
        gpu_ms(ctx, |enc| {
            encode_candidate(
                ctx,
                enc,
                &weight,
                &input,
                &cpu_tpe,
                &cpu_ids,
                &candidate_output,
                &fixture,
                expert_stride,
            )
        });
        gpu_ms(ctx, |enc| {
            encode_map(ctx, enc, &selected, &gpu_tpe, &gpu_ids, &fixture);
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            encode_candidate(
                ctx,
                enc,
                &weight,
                &input,
                &gpu_tpe,
                &gpu_ids,
                &candidate_output,
                &fixture,
                expert_stride,
            );
        });
    }

    let mut samples = Samples::new();
    for _ in 0..SAMPLES {
        samples.gather.push(gpu_ms(ctx, |enc| {
            encode_gather(ctx, enc, &input, &sorted_tokens_buf, &experts)
        }));
        samples.map.push(gpu_ms(ctx, |enc| {
            encode_map(ctx, enc, &selected, &gpu_tpe, &gpu_ids, &fixture)
        }));
        samples.existing_gate.push(gpu_ms(ctx, |enc| {
            encode_existing_gate(ctx, enc, &weight, &n, &k, &experts, expert_stride)
        }));
        samples.candidate_gate.push(gpu_ms(ctx, |enc| {
            encode_candidate(
                ctx,
                enc,
                &weight,
                &input,
                &cpu_tpe,
                &cpu_ids,
                &candidate_output,
                &fixture,
                expert_stride,
            )
        }));
        samples.existing_total.push(gpu_ms(ctx, |enc| {
            encode_gather(ctx, enc, &input, &sorted_tokens_buf, &experts);
            encode_existing_gate(ctx, enc, &weight, &n, &k, &experts, expert_stride);
        }));
        samples.candidate_total.push(gpu_ms(ctx, |enc| {
            encode_map(ctx, enc, &selected, &gpu_tpe, &gpu_ids, &fixture);
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            encode_candidate(
                ctx,
                enc,
                &weight,
                &input,
                &gpu_tpe,
                &gpu_ids,
                &candidate_output,
                &fixture,
                expert_stride,
            );
        }));
    }
    print_samples(fixture.name, &samples);
}

#[test]
#[ignore = "requires Apple Metal 4; run with --ignored --nocapture"]
fn qwen_moe_llama_gate_microbench() {
    let ctx = compute::build_metal_context().expect("Metal device required");
    if !ctx.tensorops_capable {
        eprintln!("skipping: Metal tensor operations unavailable");
        return;
    }
    // The production-size pass heats the device enough for the short reduced
    // fixture's single discarded warm-up to be representative.
    run_fixture(&ctx, production_sparse_fixture());
    run_fixture(&ctx, dense_top2_fixture());
}
