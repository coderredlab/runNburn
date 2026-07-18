//! pm35 M2: prefill GDN in_proj/gate single-output batch GEMM(M>1). FFN prefill chain 의
//! single-GEMM 축소판 — norm/residual/silu 없음. encode_gemm_q4k/q6k_tensorops(generic) 재사용.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};
use std::cell::Cell;

use crate::compute::{self, MetalContext};
pub(crate) use crate::ffn_chain::{empty_f16_buf, empty_f32_buf, readback, u32_buf};

// (호출 횟수, 누적 GPU ms) — RNB_METAL_PREFILL_GDN_INPROJ_TIME 계측용. FFN 카운터와 분리.
thread_local! {
    static PREFILL_GDN_PROJ_GPU: Cell<(usize, f64)> = const { Cell::new((0, 0.0)) };
    // pm44 D측정: host 왕복 분해 누적 (upload, commit+wait wall, readback) ms.
    static PREFILL_GDN_PROJ_HOST: Cell<(f64, f64, f64)> = const { Cell::new((0.0, 0.0, 0.0)) };
}

/// pm36: GDN proj tensorops 커널 quant 선택(Q4_K/Q5_K/Q6_K). metal crate 자체 enum으로
/// rnb-core GGMLType 의존 회피 — caller(runtime wrapper)가 GGMLType → 이 enum 변환.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensoropsQuant {
    Q4K,
    Q5K,
    Q6K,
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillProjTrace {
    pub role: &'static str,
    pub layer_idx: usize,
    pub timing_enabled: bool,
}

pub(crate) struct PrefillGdnProjCarrier {
    pub hidden_dim: usize,
    pub n_out: usize,
    pub seq_len: usize,
    pub normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*hidden]
    pub out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,    // [seq*n_out]
    pub n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,      // u32(n_out)
    pub k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,      // u32(hidden)
    pub m_buf: Retained<ProtocolObject<dyn MTLBuffer>>,      // u32(seq)
    // pm42 M4: v2 GEMM activation(matmul2d f16 강제) — normed f32 → f16 cast 1회 후 v2 GEMM input.
    pub normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*hidden] half
    pub normed_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(seq*hidden) — cast n
}

impl PrefillGdnProjCarrier {
    pub(crate) fn new(ctx: &MetalContext, hidden_dim: usize, n_out: usize, seq_len: usize) -> Self {
        Self {
            hidden_dim,
            n_out,
            seq_len,
            normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            out_dev: empty_f32_buf(ctx, seq_len * n_out),
            n_buf: u32_buf(ctx, n_out as u32),
            k_buf: u32_buf(ctx, hidden_dim as u32),
            m_buf: u32_buf(ctx, seq_len as u32),
            normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            normed_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
        }
    }
    fn upload_normed(&self, normed: &[f32]) {
        debug_assert_eq!(normed.len(), self.seq_len * self.hidden_dim);
        let c = self.normed_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(normed.as_ptr(), c.as_ptr() as *mut f32, normed.len());
        }
    }
}

pub(crate) struct PrefillGdnF32DualCarrier {
    pub hidden_dim: usize,
    pub n_out: usize,
    pub seq_len: usize,
    pub normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub left_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub right_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub m_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub normed_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl PrefillGdnF32DualCarrier {
    pub(crate) fn new(ctx: &MetalContext, hidden_dim: usize, n_out: usize, seq_len: usize) -> Self {
        Self {
            hidden_dim,
            n_out,
            seq_len,
            normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            left_out_dev: empty_f32_buf(ctx, seq_len * n_out),
            right_out_dev: empty_f32_buf(ctx, seq_len * n_out),
            n_buf: u32_buf(ctx, n_out as u32),
            k_buf: u32_buf(ctx, hidden_dim as u32),
            m_buf: u32_buf(ctx, seq_len as u32),
            normed_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
        }
    }

    fn upload_normed(&self, normed: &[f32]) {
        debug_assert_eq!(normed.len(), self.seq_len * self.hidden_dim);
        let c = self.normed_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(normed.as_ptr(), c.as_ptr() as *mut f32, normed.len());
        }
    }
}

pub(crate) struct PrefillF32ProjCarrier {
    pub hidden_dim: usize,
    pub n_out: usize,
    pub seq_len: usize,
    pub normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub m_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl PrefillF32ProjCarrier {
    pub(crate) fn new(ctx: &MetalContext, hidden_dim: usize, n_out: usize, seq_len: usize) -> Self {
        Self {
            hidden_dim,
            n_out,
            seq_len,
            normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            out_dev: empty_f32_buf(ctx, seq_len * n_out),
            n_buf: u32_buf(ctx, n_out as u32),
            k_buf: u32_buf(ctx, hidden_dim as u32),
            m_buf: u32_buf(ctx, seq_len as u32),
        }
    }

    fn upload_normed(&self, normed: &[f32]) {
        debug_assert_eq!(normed.len(), self.seq_len * self.hidden_dim);
        let c = self.normed_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(normed.as_ptr(), c.as_ptr() as *mut f32, normed.len());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_prefill_f32_proj(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n_out: usize,
    seq_len: usize,
) {
    enc.setComputePipelineState(&ctx.prefill_f32_proj_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(off_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 6);
    }
    let grid = MTLSize {
        width: seq_len.div_ceil(16),
        height: n_out.div_ceil(16),
        depth: 1,
    };
    let tg = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
fn encode_prefill_gdn_quant_proj(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillGdnF32DualCarrier,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    off: u32,
    quant: TensoropsQuant,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    input_f16: bool,
    m: usize,
) {
    if input_f16 {
        match quant {
            TensoropsQuant::Q6K => compute::encode_gemm_q6k_tensorops_v2(
                ctx,
                enc,
                w_buf,
                off,
                &carrier.normed_f16_dev,
                out_buf,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
            TensoropsQuant::Q5K => compute::encode_gemm_q5k_tensorops_v2(
                ctx,
                enc,
                w_buf,
                off,
                &carrier.normed_f16_dev,
                out_buf,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
            TensoropsQuant::Q4K => compute::encode_gemm_q4k_tensorops_v2(
                ctx,
                enc,
                w_buf,
                off,
                &carrier.normed_f16_dev,
                out_buf,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
        }
    } else {
        match quant {
            TensoropsQuant::Q6K => compute::encode_gemm_q6k_tensorops(
                ctx,
                enc,
                w_buf,
                off,
                &carrier.normed_dev,
                out_buf,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
            TensoropsQuant::Q5K => compute::encode_gemm_q5k_tensorops(
                ctx,
                enc,
                w_buf,
                off,
                &carrier.normed_dev,
                out_buf,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
            TensoropsQuant::Q4K => compute::encode_gemm_q4k_tensorops(
                ctx,
                enc,
                w_buf,
                off,
                &carrier.normed_dev,
                out_buf,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_gdn_f32_dual_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillGdnF32DualCarrier,
    normed: &[f32],
    left_w_buf: &ProtocolObject<dyn MTLBuffer>,
    left_off_buf: &ProtocolObject<dyn MTLBuffer>,
    right_w_buf: &ProtocolObject<dyn MTLBuffer>,
    right_off_buf: &ProtocolObject<dyn MTLBuffer>,
) -> (Vec<f32>, Vec<f32>) {
    carrier.upload_normed(normed);
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    encode_prefill_f32_proj(
        ctx,
        &enc,
        left_w_buf,
        left_off_buf,
        &carrier.normed_dev,
        &carrier.left_out_dev,
        &carrier.n_buf,
        &carrier.k_buf,
        &carrier.m_buf,
        carrier.n_out,
        carrier.seq_len,
    );
    encode_prefill_f32_proj(
        ctx,
        &enc,
        right_w_buf,
        right_off_buf,
        &carrier.normed_dev,
        &carrier.right_out_dev,
        &carrier.n_buf,
        &carrier.k_buf,
        &carrier.m_buf,
        carrier.n_out,
        carrier.seq_len,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    (
        readback(&carrier.left_out_dev, carrier.seq_len * carrier.n_out),
        readback(&carrier.right_out_dev, carrier.seq_len * carrier.n_out),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_gdn_quant_dual_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillGdnF32DualCarrier,
    normed: &[f32],
    left_w_buf: &ProtocolObject<dyn MTLBuffer>,
    left_off_buf: &ProtocolObject<dyn MTLBuffer>,
    left_quant: TensoropsQuant,
    right_w_buf: &ProtocolObject<dyn MTLBuffer>,
    right_off_buf: &ProtocolObject<dyn MTLBuffer>,
    right_quant: TensoropsQuant,
    m: usize,
) -> (Vec<f32>, Vec<f32>) {
    carrier.upload_normed(normed);
    let left_off = unsafe { *(left_off_buf.contents().as_ptr() as *const u32) };
    let right_off = unsafe { *(right_off_buf.contents().as_ptr() as *const u32) };
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    let left_v2_ready = match left_quant {
        TensoropsQuant::Q4K => ctx.gemm_q4k_tensorops_v2_pipeline.is_some(),
        TensoropsQuant::Q5K => ctx.gemm_q5k_tensorops_v2_pipeline.is_some(),
        TensoropsQuant::Q6K => ctx.gemm_q6k_tensorops_v2_pipeline.is_some(),
    };
    let right_v2_ready = match right_quant {
        TensoropsQuant::Q4K => ctx.gemm_q4k_tensorops_v2_pipeline.is_some(),
        TensoropsQuant::Q5K => ctx.gemm_q5k_tensorops_v2_pipeline.is_some(),
        TensoropsQuant::Q6K => ctx.gemm_q6k_tensorops_v2_pipeline.is_some(),
    };
    let use_v2 = ctx.tensorops_capable
        && ctx.cast_f32_f16_pipeline.is_some()
        && left_v2_ready
        && right_v2_ready
        && std::env::var("RNB_METAL_PREFILL_GDN_PROJ_V2").as_deref() != Ok("0");
    if use_v2 {
        compute::encode_cast_f32_to_f16(
            ctx,
            &enc,
            &carrier.normed_dev,
            &carrier.normed_f16_dev,
            &carrier.normed_elems_buf,
            m * carrier.hidden_dim,
        );
    }
    encode_prefill_gdn_quant_proj(
        ctx,
        &enc,
        carrier,
        left_w_buf,
        left_off,
        left_quant,
        &carrier.left_out_dev,
        use_v2,
        m,
    );
    encode_prefill_gdn_quant_proj(
        ctx,
        &enc,
        carrier,
        right_w_buf,
        right_off,
        right_quant,
        &carrier.right_out_dev,
        use_v2,
        m,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    (
        readback(&carrier.left_out_dev, m * carrier.n_out),
        readback(&carrier.right_out_dev, m * carrier.n_out),
    )
}

pub(crate) fn prefill_f32_proj_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillF32ProjCarrier,
    normed: &[f32],
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
) -> Vec<f32> {
    carrier.upload_normed(normed);
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    encode_prefill_f32_proj(
        ctx,
        &enc,
        w_buf,
        off_buf,
        &carrier.normed_dev,
        &carrier.out_dev,
        &carrier.n_buf,
        &carrier.k_buf,
        &carrier.m_buf,
        carrier.n_out,
        carrier.seq_len,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    readback(&carrier.out_dev, carrier.seq_len * carrier.n_out)
}

pub(crate) fn prefill_f32_proj_encode_no_readback(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillF32ProjCarrier,
    normed: &[f32],
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    carrier.upload_normed(normed);
    encode_prefill_f32_proj(
        ctx,
        enc,
        w_buf,
        off_buf,
        &carrier.normed_dev,
        &carrier.out_dev,
        &carrier.n_buf,
        &carrier.k_buf,
        &carrier.m_buf,
        carrier.n_out,
        carrier.seq_len,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_gdn_proj_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillGdnProjCarrier,
    normed: &[f32],
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    quant: TensoropsQuant,
    m: usize,
    trace: Option<PrefillProjTrace>,
) -> Vec<f32> {
    let t_total0 = std::time::Instant::now();
    let t_up0 = std::time::Instant::now();
    carrier.upload_normed(normed);
    let t_up = t_up0.elapsed().as_secs_f64() * 1000.0;
    let off = unsafe { *(off_buf.contents().as_ptr() as *const u32) };
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    // pm42 M4: v2 GEMM(llama cooperative C 64×128, FFN pm42 9.0x 검증). activation matmul2d f16
    // 강제라 normed f32 → f16 cast 1회 후 v2 GEMM(GDN in_proj/gate/ssm_out + ATN qkv/o, generic seam).
    // **default ON 승격**(pm43: GDN/ATN proj GPU 6.58→0.79s = −88.0%/8.3x, prefill 16.3→10.48s,
    // 27B token-identical). RNB_METAL_PREFILL_GDN_PROJ_V2=0 으로 v1 opt-out.
    let v2_pipeline_ready = match quant {
        TensoropsQuant::Q4K => ctx.gemm_q4k_tensorops_v2_pipeline.is_some(),
        TensoropsQuant::Q5K => ctx.gemm_q5k_tensorops_v2_pipeline.is_some(),
        TensoropsQuant::Q6K => ctx.gemm_q6k_tensorops_v2_pipeline.is_some(),
    };
    let use_v2 = ctx.tensorops_capable
        && ctx.cast_f32_f16_pipeline.is_some()
        && v2_pipeline_ready
        && std::env::var("RNB_METAL_PREFILL_GDN_PROJ_V2").as_deref() != Ok("0");
    if use_v2 {
        // normed f32 → f16 (1회). 이후 v2 GEMM 이 normed_f16_dev 를 device-direct tensor 로.
        compute::encode_cast_f32_to_f16(
            ctx,
            &enc,
            &carrier.normed_dev,
            &carrier.normed_f16_dev,
            &carrier.normed_elems_buf,
            m * carrier.hidden_dim,
        );
        match quant {
            TensoropsQuant::Q6K => compute::encode_gemm_q6k_tensorops_v2(
                ctx,
                &enc,
                w_buf,
                off,
                &carrier.normed_f16_dev,
                &carrier.out_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
            TensoropsQuant::Q5K => compute::encode_gemm_q5k_tensorops_v2(
                ctx,
                &enc,
                w_buf,
                off,
                &carrier.normed_f16_dev,
                &carrier.out_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
            TensoropsQuant::Q4K => compute::encode_gemm_q4k_tensorops_v2(
                ctx,
                &enc,
                w_buf,
                off,
                &carrier.normed_f16_dev,
                &carrier.out_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
        }
    } else {
        match quant {
            TensoropsQuant::Q6K => compute::encode_gemm_q6k_tensorops(
                ctx,
                &enc,
                w_buf,
                off,
                &carrier.normed_dev,
                &carrier.out_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
            TensoropsQuant::Q5K => compute::encode_gemm_q5k_tensorops(
                ctx,
                &enc,
                w_buf,
                off,
                &carrier.normed_dev,
                &carrier.out_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
            TensoropsQuant::Q4K => compute::encode_gemm_q4k_tensorops(
                ctx,
                &enc,
                w_buf,
                off,
                &carrier.normed_dev,
                &carrier.out_dev,
                &carrier.n_buf,
                &carrier.k_buf,
                &carrier.m_buf,
                carrier.n_out,
                m,
            ),
        }
    }
    enc.endEncoding();
    let t_cw0 = std::time::Instant::now();
    cmd.commit();
    cmd.waitUntilCompleted();
    let t_cw = t_cw0.elapsed().as_secs_f64() * 1000.0;
    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
    let t_rb0 = std::time::Instant::now();
    let contents = carrier.out_dev.contents();
    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, m * carrier.n_out) };
    let result = out.to_vec();
    let t_rb = t_rb0.elapsed().as_secs_f64() * 1000.0;
    let t_total = t_total0.elapsed().as_secs_f64() * 1000.0;
    if std::env::var("RNB_METAL_PREFILL_GDN_INPROJ_TIME").as_deref() == Ok("1") {
        let (cnt, gsum) = PREFILL_GDN_PROJ_GPU.with(|c| {
            let (cnt, sum) = c.get();
            c.set((cnt + 1, sum + gpu_ms));
            (cnt + 1, sum + gpu_ms)
        });
        // pm44 D측정: host 왕복 = upload + (commit+wait wall) + readback. launch_overhead = cw_wall - gpu.
        PREFILL_GDN_PROJ_HOST.with(|c| {
            let (u, w, r) = c.get();
            let (uu, ww, rr) = (u + t_up, w + t_cw, r + t_rb);
            c.set((uu, ww, rr));
            eprintln!(
                "[prefill-gdn-proj] call#{cnt} quant={quant:?} n_out={} m={m} | gpu={gpu_ms:.3} cw_wall={t_cw:.3} up={t_up:.3} rb={t_rb:.3} || cumul gpu={gsum:.1} up={uu:.1} cw_wall={ww:.1} rb={rr:.1} launch_overhead(cw-gpu)={:.1}ms",
                carrier.n_out,
                ww - gsum,
            );
        });
    }
    if let Some(trace) = trace.filter(|trace| trace.timing_enabled) {
        eprintln!(
            "[prefill-atn-proj-time] layer={} role={} quant={quant:?} n_out={} m={m} upload_ms={t_up:.3} gpu_ms={gpu_ms:.3} commit_wait_ms={t_cw:.3} readback_ms={t_rb:.3} total_ms={t_total:.3}",
            trace.layer_idx,
            trace.role,
            carrier.n_out,
        );
    }
    result
}

/// GDN one-shot prefill projection. All inputs and outputs remain device-resident;
/// this function only appends work to the caller-owned encoder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_gdn_projection(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weight: crate::GdnBackendWeightRef<'_>,
    input_f32: &ProtocolObject<dyn MTLBuffer>,
    input_f16: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    output_byte_offset: usize,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
    seq_len: usize,
) -> Result<(), String> {
    match weight {
        crate::GdnBackendWeightRef::F32 {
            values,
            rows: weight_rows,
            cols: weight_cols,
        } => {
            if weight_rows != rows || weight_cols != cols || values.len() != rows * cols {
                return Err("GDN F32 projection shape mismatch".to_string());
            }
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            let (weight_buf, weight_offset) = compute::wrap_nocopy(ctx, bytes);
            let weight_offset_buf = u32_buf(ctx, weight_offset);
            enc.setComputePipelineState(&ctx.prefill_f32_proj_pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&weight_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(input_f32), 0, 1);
                enc.setBuffer_offset_atIndex(Some(output), output_byte_offset, 2);
                enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&weight_offset_buf), 0, 5);
                enc.setBuffer_offset_atIndex(Some(m_buf), 0, 6);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: seq_len.div_ceil(16),
                    height: rows.div_ceil(16),
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }
        crate::GdnBackendWeightRef::Quant(view) => {
            if view.rows != rows || view.cols != cols {
                return Err("GDN quant projection shape mismatch".to_string());
            }
            let pipeline = match view.quant {
                TensoropsQuant::Q4K => ctx.gemm_q4k_tensorops_v2_pipeline.as_ref(),
                TensoropsQuant::Q5K => ctx.gemm_q5k_tensorops_v2_pipeline.as_ref(),
                TensoropsQuant::Q6K => ctx.gemm_q6k_tensorops_v2_pipeline.as_ref(),
            }
            .ok_or_else(|| "GDN quant projection pipeline unavailable".to_string())?;
            if !ctx.tensorops_capable || ctx.cast_f32_f16_pipeline.is_none() {
                return Err("GDN quant projection tensorops unavailable".to_string());
            }
            let (weight_buf, weight_offset) = compute::wrap_nocopy(ctx, view.raw);
            enc.setComputePipelineState(pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&weight_buf), weight_offset as usize, 0);
                enc.setBuffer_offset_atIndex(Some(input_f16), 0, 1);
                enc.setBuffer_offset_atIndex(Some(output), output_byte_offset, 2);
                enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(k_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(m_buf), 0, 5);
                enc.setThreadgroupMemoryLength_atIndex(
                    match view.quant {
                        TensoropsQuant::Q6K => 64 * 128 * 2,
                        TensoropsQuant::Q4K | TensoropsQuant::Q5K => 64 * 64 * 2,
                    },
                    0,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: seq_len.div_ceil(128),
                    height: rows.div_ceil(64),
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }
    Ok(())
}
