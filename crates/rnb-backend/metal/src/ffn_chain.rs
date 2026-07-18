use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLArgumentEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLDevice, MTLResource, MTLResourceOptions, MTLResourceUsage,
    MTLSize, MTLStorageMode, MTLTimestamp,
};

use crate::compute::MetalContext;
use crate::{QwenMoePrefillBackendSpecRef, QwenRouteAlgorithm};

const QWEN_MOE_WEIGHT_TABLE_CAP: usize = 257;
const QWEN_MOE_WEIGHT_CHUNK_EXPERTS: usize = 32;

/// RMSNorm reduction threadgroup 크기. 2의 거듭제곱(tree reduction 정확) +
/// SIMD width 32 의 배수. grid-stride 라 dim 보다 작아도 정확.
/// ⚠️ rms_norm.metal 의 `threadgroup float partial[256]` 배열 크기와 **반드시 일치**해야 한다.
/// 이 값을 바꾸면 MSL 배열 크기도 같이 바꿔야 함(안 그러면 tree reduction 이 범위 밖 read).
const RMS_TG_SIZE: usize = 256;

fn qwen_moe_prefill_gather_f16_requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_GATHER_F16")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true)
    })
}

fn qwen_moe_prefill_down_scatter_requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER")
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true)
    })
}

/// f32 host slice → StorageModeShared MTLBuffer (복사 업로드).
pub(crate) fn shared_f32_buf(
    ctx: &MetalContext,
    data: &[f32],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bytes = std::mem::size_of_val(data);
    unsafe {
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("f32 ptr null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared)
            .expect("shared f32 buffer")
    }
}

/// u32 host slice → StorageModeShared MTLBuffer (복사 업로드).
pub(crate) fn shared_u32_buf(
    ctx: &MetalContext,
    data: &[u32],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bytes = std::mem::size_of_val(data);
    unsafe {
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("u32 ptr null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared)
            .expect("shared u32 buffer")
    }
}

/// u32 scalar → MTLBuffer.
pub(crate) fn u32_buf(ctx: &MetalContext, v: u32) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        let ptr = NonNull::new(&v as *const u32 as *mut std::ffi::c_void).expect("u32 ptr null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, 4, MTLResourceOptions::StorageModeShared)
            .expect("u32 buffer")
    }
}

/// u64 scalar → MTLBuffer.
pub(crate) fn u64_buf(ctx: &MetalContext, v: u64) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        let ptr = NonNull::new(&v as *const u64 as *mut std::ffi::c_void).expect("u64 ptr null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, 8, MTLResourceOptions::StorageModeShared)
            .expect("u64 buffer")
    }
}

/// f32 scalar → MTLBuffer.
pub(crate) fn f32_buf(ctx: &MetalContext, v: f32) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        let ptr = NonNull::new(&v as *const f32 as *mut std::ffi::c_void).expect("f32 ptr null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, 4, MTLResourceOptions::StorageModeShared)
            .expect("f32 scalar buffer")
    }
}

fn set_u32_bytes(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: u32, index: usize) {
    unsafe {
        let ptr = NonNull::new(&value as *const u32 as *mut std::ffi::c_void)
            .expect("u32 bytes ptr null");
        enc.setBytes_length_atIndex(ptr, std::mem::size_of::<u32>(), index);
    }
}

fn set_f32_bytes(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: f32, index: usize) {
    unsafe {
        let ptr = NonNull::new(&value as *const f32 as *mut std::ffi::c_void)
            .expect("f32 bytes ptr null");
        enc.setBytes_length_atIndex(ptr, std::mem::size_of::<f32>(), index);
    }
}

/// u8 host slice → StorageModeShared MTLBuffer (복사).
fn shared_u8_buf(ctx: &MetalContext, data: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("u8 ptr null");
        ctx.device
            .newBufferWithBytes_length_options(
                ptr,
                data.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("shared u8 buffer")
    }
}

/// 빈 device buffer(len f32) — 결과 쓰기용.
pub(crate) fn empty_f32_buf(
    ctx: &MetalContext,
    len: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    ctx.device
        .newBufferWithLength_options(len * 4, MTLResourceOptions::StorageModeShared)
        .expect("empty f32 buffer")
}

pub(crate) fn private_f32_buf(
    ctx: &MetalContext,
    len: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    ctx.device
        .newBufferWithLength_options(len * 4, MTLResourceOptions::StorageModePrivate)
        .expect("private f32 buffer")
}

/// 빈 device buffer(len f16=2byte) — pm42 M3 v2 GEMM activation(matmul2d f16 강제) 용.
pub(crate) fn empty_f16_buf(
    ctx: &MetalContext,
    len: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    ctx.device
        .newBufferWithLength_options(len * 2, MTLResourceOptions::StorageModeShared)
        .expect("empty f16 buffer")
}

pub(crate) fn private_f16_buf(
    ctx: &MetalContext,
    len: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    ctx.device
        .newBufferWithLength_options(len * 2, MTLResourceOptions::StorageModePrivate)
        .expect("private f16 buffer")
}

/// MTLBuffer contents → Vec<f32> readback (len 개).
pub(crate) fn readback(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<f32> {
    let contents = buf.contents();
    unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, len).to_vec() }
}

/// RMSNorm 을 주어진 compute encoder 에 encode (commit 안 함).
/// in/weight → out, 단일 threadgroup(grid=1, tg=RMS_TG_SIZE).
pub(crate) fn encode_rms_norm(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    dim_buf: &ProtocolObject<dyn MTLBuffer>,
    eps_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    enc.setComputePipelineState(&ctx.rms_norm_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dim_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(eps_buf), 0, 4);
    }
    let grid = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: RMS_TG_SIZE,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// SwiGLU in-place(gate = silu(gate)*up) 를 encoder 에 encode.
/// thread 당 1 element (grid = dim, tg = threadExecutionWidth).
pub(crate) fn encode_silu_mul(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    up_buf: &ProtocolObject<dyn MTLBuffer>,
    dim_buf: &ProtocolObject<dyn MTLBuffer>,
    dim: usize,
) {
    enc.setComputePipelineState(&ctx.silu_mul_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dim_buf), 0, 2);
    }
    let tg_width = ctx.silu_mul_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: dim.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// residual in-place(hidden += down) 를 encoder 에 encode.
pub(crate) fn encode_residual_add(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    hidden_buf: &ProtocolObject<dyn MTLBuffer>,
    down_buf: &ProtocolObject<dyn MTLBuffer>,
    dim_buf: &ProtocolObject<dyn MTLBuffer>,
    dim: usize,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(dim_buf), 0, 2);
    }
    encode_residual_add_bound(ctx, enc, hidden_buf, down_buf, dim);
}

fn encode_residual_add_inline_dim(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    hidden_buf: &ProtocolObject<dyn MTLBuffer>,
    down_buf: &ProtocolObject<dyn MTLBuffer>,
    dim: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    let dim_u32 = u32::try_from(dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    set_u32_bytes(enc, dim_u32, 2);
    encode_residual_add_bound(ctx, enc, hidden_buf, down_buf, dim);
    Ok(())
}

fn encode_residual_add_bound(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    hidden_buf: &ProtocolObject<dyn MTLBuffer>,
    down_buf: &ProtocolObject<dyn MTLBuffer>,
    dim: usize,
) {
    enc.setComputePipelineState(&ctx.residual_add_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(hidden_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(down_buf), 0, 1);
    }
    let tg_width = ctx.residual_add_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: dim.div_ceil(tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// FFN chain 의 device-resident 중간 버퍼 + scalar 상수 버퍼.
/// shape(hidden_dim, ffn_dim) 별 1회 alloc 후 재사용. `!Send+!Sync` 라 thread_local.
pub(crate) struct FfnCarrier {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub hdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub fdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // gate/up GEMV 의 K = hidden_dim
    pub k_ffn_buf: Retained<ProtocolObject<dyn MTLBuffer>>,    // down GEMV 의 K = ffn_dim
}

impl FfnCarrier {
    pub(crate) fn new(ctx: &MetalContext, hidden_dim: usize, ffn_dim: usize, eps: f32) -> Self {
        Self {
            hidden_dim,
            ffn_dim,
            hidden_dev: empty_f32_buf(ctx, hidden_dim),
            normed_dev: empty_f32_buf(ctx, hidden_dim),
            gate_dev: empty_f32_buf(ctx, ffn_dim),
            up_dev: empty_f32_buf(ctx, ffn_dim),
            down_dev: empty_f32_buf(ctx, hidden_dim),
            hdim_buf: u32_buf(ctx, hidden_dim as u32),
            fdim_buf: u32_buf(ctx, ffn_dim as u32),
            eps_buf: f32_buf(ctx, eps),
            k_hidden_buf: u32_buf(ctx, hidden_dim as u32),
            k_ffn_buf: u32_buf(ctx, ffn_dim as u32),
        }
    }

    /// hidden host slice → hidden_dev 업로드 (StorageModeShared contents 직접 쓰기).
    fn upload_hidden(&self, hidden: &[f32]) {
        debug_assert_eq!(hidden.len(), self.hidden_dim);
        let contents = self.hidden_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                contents.as_ptr() as *mut f32,
                self.hidden_dim,
            );
        }
    }
}

/// FFN chain 한 token 실행. weight buffer/offset(NoCopy resident)은 caller 가 준비.
/// gate/up: N=ffn_dim, K=hidden_dim. down: N=hidden_dim, K=ffn_dim.
/// carrier 의 device buffer 를 재사용하며, 단일 command buffer 단일 encoder 로 ①~⑥.
/// 반환: residual 적용된 hidden(hidden_dim).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_chain_dispatch(
    ctx: &MetalContext,
    carrier: &FfnCarrier,
    hidden: &[f32],
    norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    down_is_q6k: bool,
) -> Vec<f32> {
    let hidden_dim = carrier.hidden_dim;
    let ffn_dim = carrier.ffn_dim;
    carrier.upload_hidden(hidden);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");

    // ① rms_norm: hidden_dev → normed_dev
    encode_rms_norm(
        ctx,
        &enc,
        &carrier.hidden_dev,
        norm_w_buf,
        &carrier.normed_dev,
        &carrier.hdim_buf,
        &carrier.eps_buf,
    );
    // ② gate: normed_dev → gate_dev  (N=ffn_dim, K=hidden_dim)
    crate::compute::encode_gemv_q4k(
        ctx,
        &enc,
        gate_w_buf,
        &carrier.normed_dev,
        &carrier.gate_dev,
        &carrier.fdim_buf,
        &carrier.k_hidden_buf,
        gate_off_buf,
        ffn_dim,
    );
    // ③ up: normed_dev → up_dev
    crate::compute::encode_gemv_q4k(
        ctx,
        &enc,
        up_w_buf,
        &carrier.normed_dev,
        &carrier.up_dev,
        &carrier.fdim_buf,
        &carrier.k_hidden_buf,
        up_off_buf,
        ffn_dim,
    );
    // ④ silu_mul: gate_dev = silu(gate_dev)*up_dev (in-place)
    encode_silu_mul(
        ctx,
        &enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.fdim_buf,
        ffn_dim,
    );
    // ⑤ down: gate_dev → down_dev (N=hidden_dim, K=ffn_dim).
    //    down weight quant 에 맞는 GEMV 커널 선택(Q4_K 144B / Q6_K 210B).
    if down_is_q6k {
        crate::compute::encode_gemv_q6k(
            ctx,
            &enc,
            down_w_buf,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.hdim_buf,
            &carrier.k_ffn_buf,
            down_off_buf,
            hidden_dim,
        );
    } else {
        crate::compute::encode_gemv_q4k(
            ctx,
            &enc,
            down_w_buf,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.hdim_buf,
            &carrier.k_ffn_buf,
            down_off_buf,
            hidden_dim,
        );
    }
    // ⑥ residual_add: hidden_dev += down_dev (in-place)
    encode_residual_add(
        ctx,
        &enc,
        &carrier.hidden_dev,
        &carrier.down_dev,
        &carrier.hdim_buf,
        hidden_dim,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    readback(&carrier.hidden_dev, hidden_dim)
}

// ---------------------------------------------------------------------------
// prefill FFN chain (M>1) — FfnCarrier/ffn_chain_dispatch 의 batch 아날로그.
// pm33. norm/residual 제외(caller 가 normed 제공 + residual 처리), GEMV→batch GEMM.
// ---------------------------------------------------------------------------

pub(crate) struct PrefillFfnCarrier {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub seq_len: usize,
    pub normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq_len * hidden_dim]
    pub gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,   // [seq_len * ffn_dim]
    pub up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,     // [seq_len * ffn_dim]
    pub down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,   // [seq_len * hidden_dim]
    pub hdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,   // u32(hidden_dim) — down GEMM 의 N
    pub fdim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,   // u32(ffn_dim) — gate/up GEMM 의 N
    pub k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(hidden_dim) — gate/up GEMM 의 K
    pub k_ffn_buf: Retained<ProtocolObject<dyn MTLBuffer>>,    // u32(ffn_dim) — down GEMM 의 K
    pub m_buf: Retained<ProtocolObject<dyn MTLBuffer>>,        // u32(seq_len) — batch M
    pub act_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(seq_len*ffn_dim) — silu element 수 + act cast n
    // pm42 M3 step2: v2 GEMM activation(f16 device-direct) — chain 에서 1회 cast 후 gate/up/down 공유.
    pub normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq_len*hidden_dim] half — gate/up input
    pub act_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq_len*ffn_dim] half — down input(silu 결과)
    pub normed_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(seq_len*hidden_dim) — normed cast n
}

pub(crate) struct QwenMoePrefillAccumCarrier {
    hidden_dim: usize,
    seq_len: usize,
    pub out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub norm_all_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct QwenMoePrefillIdCarrier {
    slots: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_ff_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct QwenMoePrefillIdGateUpCarrier {
    slots: usize,
    ffn_dim: usize,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct QwenMoePrefillIdGateUpF16Carrier {
    slots: usize,
    ffn_dim: usize,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct QwenMoeMulMmIdV3Carrier {
    pub(crate) slots: usize,
    pub(crate) seq_len: usize,
    pub(crate) n_expert_used: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) ffn_dim: usize,
    gate_half_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_half_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_token_rank_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch: QwenMoeMulMmIdV3Scratch,
}

pub(crate) struct QwenMoeMulMmIdV4Carrier {
    pub(crate) slots: usize,
    pub(crate) n_expert_used: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) ffn_dim: usize,
    pub(crate) gate_half_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) up_half_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch: QwenMoeMulMmIdV4Scratch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenMoeIdPreflightReason {
    PipelineUnavailable,
    InvalidExpertArena,
    ExpertArenaMismatch,
    ExpertIdOutOfRange,
    TokenIdOutOfRange,
    UnsortedExpertIds,
    DuplicateTokenInExpertBlock,
    UnsupportedDownQuant,
    InvalidPrimitiveShape,
    BlockCountOverflow,
    PartialBufferOverBudget,
    RankCountMismatch,
    ScratchOverBudget,
    DispatchCountOverBudget,
}

impl QwenMoeIdPreflightReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::PipelineUnavailable => "pipeline_unavailable",
            Self::InvalidExpertArena => "invalid_expert_arena",
            Self::ExpertArenaMismatch => "expert_arena_mismatch",
            Self::ExpertIdOutOfRange => "expert_id_out_of_range",
            Self::TokenIdOutOfRange => "token_id_out_of_range",
            Self::UnsortedExpertIds => "unsorted_expert_ids",
            Self::DuplicateTokenInExpertBlock => "duplicate_token_in_expert_block",
            Self::UnsupportedDownQuant => "unsupported_down_quant",
            Self::InvalidPrimitiveShape => "invalid_primitive_shape",
            Self::BlockCountOverflow => "block_count_overflow",
            Self::PartialBufferOverBudget => "partial_buffer_over_budget",
            Self::RankCountMismatch => "rank_count_mismatch",
            Self::ScratchOverBudget => "scratch_over_budget",
            Self::DispatchCountOverBudget => "dispatch_count_over_budget",
        }
    }

    pub(crate) fn as_mulmmid_v3_str(self) -> &'static str {
        match self {
            Self::UnsupportedDownQuant => "unsupported_quant",
            Self::InvalidPrimitiveShape => "invalid_shape",
            Self::PartialBufferOverBudget | Self::ScratchOverBudget => "scratch_over_budget",
            other => other.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QwenMoeIdExecutionPlan {
    pub(crate) expert_offsets: Vec<u32>,
    pub(crate) expert_counts: Vec<u32>,
    pub(crate) block_experts: Vec<u32>,
    pub(crate) block_local0: Vec<u32>,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) route_weights: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QwenMoeMulMmIdV3Plan {
    pub(crate) expert_offsets: Vec<u32>,
    pub(crate) expert_counts: Vec<u32>,
    pub(crate) block_experts: Vec<u32>,
    pub(crate) block_local0: Vec<u32>,
    pub(crate) token_ids_sorted: Vec<u32>,
    pub(crate) rank_ids_sorted: Vec<u32>,
    pub(crate) dst_slots_sorted: Vec<u32>,
    pub(crate) route_weights_sorted: Vec<f32>,
    pub(crate) route_weights_token_rank: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QwenMoeMulMmIdV4Plan {
    pub(crate) expert_offsets: Vec<u32>,
    pub(crate) expert_counts: Vec<u32>,
    pub(crate) id_block_experts: Vec<u32>,
    pub(crate) id_block_local0: Vec<u32>,
    pub(crate) expert_rank_offsets: Vec<u32>,
    pub(crate) expert_rank_counts: Vec<u32>,
    pub(crate) rank_block_offsets: Vec<u32>,
    pub(crate) rank_block_counts: Vec<u32>,
    pub(crate) rank_block_experts: Vec<u32>,
    pub(crate) rank_block_local0: Vec<u32>,
    pub(crate) token_ids_sorted: Vec<u32>,
    pub(crate) rank_ids_sorted: Vec<u32>,
    pub(crate) route_weights_sorted: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QwenMoeRawIdMap {
    pub(crate) expert_counts: Vec<u32>,
    pub(crate) route_slot_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenMoeMulMmIdV3Scratch {
    pub(crate) gate_bytes: usize,
    pub(crate) up_bytes: usize,
    pub(crate) down_bytes: usize,
    pub(crate) metadata_bytes: usize,
    pub(crate) total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenMoeMulMmIdV4Scratch {
    pub(crate) gate_bytes: usize,
    pub(crate) up_bytes: usize,
    pub(crate) down_bytes: usize,
    pub(crate) metadata_bytes: usize,
    pub(crate) total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeLlamaIdQuant {
    Q4K,
    Q5K,
    Q6K,
    Q8Zero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMoeLlamaIdInput {
    F32,
    F16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenMoeLlamaIdQuantSet {
    pub gate: QwenMoeLlamaIdQuant,
    pub up: QwenMoeLlamaIdQuant,
    pub down: QwenMoeLlamaIdQuant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenMoeLlamaIdMatmulShape {
    pub input_dim: usize,
    pub output_dim: usize,
    pub n_tokens: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_weight_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QwenMoeLlamaIdTrace {
    pub map_dispatches: usize,
    pub gate_dispatches: usize,
    pub up_dispatches: usize,
    pub activation_dispatches: usize,
    pub down_dispatches: usize,
    pub reduce_dispatches: usize,
    pub map_ms: Option<f64>,
    pub gate_ms: Option<f64>,
    pub up_ms: Option<f64>,
    pub activation_ms: Option<f64>,
    pub down_ms: Option<f64>,
    pub reduce_ms: Option<f64>,
    pub stage_timing_source: Option<&'static str>,
    pub stage_timing_unsupported_reason: Option<&'static str>,
    pub stage_timing_fallback_reason: Option<&'static str>,
    pub scratch_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenMoeLlamaIdCounterCalibration {
    gpu_timestamp: MTLTimestamp,
    wall: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenMoeLlamaIdStage {
    Map,
    Gate,
    Up,
    Activation,
    Down,
    Reduce,
}

#[derive(Debug, Clone, Copy)]
struct QwenMoeLlamaIdStageInterval {
    stage: QwenMoeLlamaIdStage,
    start_sample: usize,
    end_sample: usize,
}

pub(crate) struct QwenMoeLlamaIdStageSampler {
    sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    next_sample: usize,
    open_stage: Option<(QwenMoeLlamaIdStage, usize)>,
    intervals: Vec<QwenMoeLlamaIdStageInterval>,
}

impl QwenMoeLlamaIdStageSampler {
    const SAMPLE_CAPACITY: usize = 32;

    pub(crate) fn new(ctx: &MetalContext) -> Result<Self, &'static str> {
        let counter_set = crate::compute::qwen_moe_llama_id_timestamp_counter_set(ctx)?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&counter_set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        unsafe {
            descriptor.setSampleCount(Self::SAMPLE_CAPACITY);
        }
        let sample_buffer = ctx
            .device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|_| "counter_sample_buffer_creation_failed")?;
        Ok(Self {
            sample_buffer,
            next_sample: 0,
            open_stage: None,
            intervals: Vec::with_capacity(Self::SAMPLE_CAPACITY / 2),
        })
    }

    pub(crate) fn calibration(ctx: &MetalContext) -> QwenMoeLlamaIdCounterCalibration {
        let mut cpu_timestamp = 0;
        let mut gpu_timestamp = 0;
        unsafe {
            ctx.device.sampleTimestamps_gpuTimestamp(
                NonNull::from(&mut cpu_timestamp),
                NonNull::from(&mut gpu_timestamp),
            );
        }
        QwenMoeLlamaIdCounterCalibration {
            gpu_timestamp,
            wall: std::time::Instant::now(),
        }
    }

    fn begin(
        &mut self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        stage: QwenMoeLlamaIdStage,
    ) {
        debug_assert!(self.open_stage.is_none());
        let sample = self.next_sample;
        self.next_sample += 1;
        unsafe {
            enc.sampleCountersInBuffer_atSampleIndex_withBarrier(&self.sample_buffer, sample, true);
        }
        self.open_stage = Some((stage, sample));
    }

    fn end(&mut self, enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        let (stage, start_sample) = self.open_stage.take().expect("stage sample begin");
        let end_sample = self.next_sample;
        self.next_sample += 1;
        unsafe {
            enc.sampleCountersInBuffer_atSampleIndex_withBarrier(
                &self.sample_buffer,
                end_sample,
                true,
            );
        }
        self.intervals.push(QwenMoeLlamaIdStageInterval {
            stage,
            start_sample,
            end_sample,
        });
    }

    pub(crate) fn resolve(
        &self,
        start: QwenMoeLlamaIdCounterCalibration,
        end: QwenMoeLlamaIdCounterCalibration,
        trace: &mut QwenMoeLlamaIdTrace,
    ) -> Result<(), &'static str> {
        if self.open_stage.is_some() || self.next_sample == 0 {
            return Err("counter_sample_sequence_incomplete");
        }
        let gpu_ticks = end
            .gpu_timestamp
            .checked_sub(start.gpu_timestamp)
            .filter(|ticks| *ticks > 0)
            .ok_or("counter_calibration_invalid")?;
        let ms_per_gpu_tick =
            end.wall.duration_since(start.wall).as_secs_f64() * 1000.0 / gpu_ticks as f64;
        if !ms_per_gpu_tick.is_finite() || ms_per_gpu_tick <= 0.0 {
            return Err("counter_calibration_invalid");
        }
        let data = unsafe {
            self.sample_buffer
                .resolveCounterRange(NSRange::new(0, self.next_sample))
        }
        .ok_or("counter_resolve_failed")?;
        let bytes = unsafe { data.as_bytes_unchecked() };
        let expected_bytes = self
            .next_sample
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or("counter_resolve_invalid")?;
        if bytes.len() < expected_bytes {
            return Err("counter_resolve_invalid");
        }
        let timestamps: Vec<u64> = bytes[..expected_bytes]
            .chunks_exact(std::mem::size_of::<u64>())
            .map(|bytes| u64::from_ne_bytes(bytes.try_into().expect("timestamp width")))
            .collect();

        let mut totals = [0.0f64; 6];
        for interval in &self.intervals {
            let begin = timestamps[interval.start_sample];
            let end = timestamps[interval.end_sample];
            let ticks = end
                .checked_sub(begin)
                .filter(|ticks| *ticks > 0 && end != u64::MAX && begin != u64::MAX)
                .ok_or("counter_timestamp_invalid")?;
            let stage_index = match interval.stage {
                QwenMoeLlamaIdStage::Map => 0,
                QwenMoeLlamaIdStage::Gate => 1,
                QwenMoeLlamaIdStage::Up => 2,
                QwenMoeLlamaIdStage::Activation => 3,
                QwenMoeLlamaIdStage::Down => 4,
                QwenMoeLlamaIdStage::Reduce => 5,
            };
            totals[stage_index] += ticks as f64 * ms_per_gpu_tick;
        }
        if totals
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("counter_stage_missing");
        }
        trace.map_ms = Some(totals[0]);
        trace.gate_ms = Some(totals[1]);
        trace.up_ms = Some(totals[2]);
        trace.activation_ms = Some(totals[3]);
        trace.down_ms = Some(totals[4]);
        trace.reduce_ms = Some(totals[5]);
        trace.stage_timing_source = Some("metal_counter_timestamp");
        trace.stage_timing_unsupported_reason = None;
        Ok(())
    }
}

fn qwen_moe_llama_id_stage_begin(
    sampler: &mut Option<&mut QwenMoeLlamaIdStageSampler>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    stage: QwenMoeLlamaIdStage,
) {
    if let Some(sampler) = sampler.as_deref_mut() {
        sampler.begin(enc, stage);
    }
}

fn qwen_moe_llama_id_stage_end(
    sampler: &mut Option<&mut QwenMoeLlamaIdStageSampler>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
) {
    if let Some(sampler) = sampler.as_deref_mut() {
        sampler.end(enc);
    }
}

pub(crate) const QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT: usize = 256;
const QWEN_MOE_LLAMA_ID_PHYSICAL_BUFFER_ALIGNMENT: usize = 4096;
const QWEN_MOE_LLAMA_Q8K_BLOCK_BYTES: usize = 292;
const QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD: usize = 1024;
const QWEN_MOE_LLAMA_ID_BLOCK_TILE: usize = 64;
const QWEN_MOE_LLAMA_ID_INDIRECT_ARGS_BYTES: usize = 32;
const QWEN_MOE_LLAMA_ID_FFN_Q4_INDIRECT_OFFSET: usize = 0;
const QWEN_MOE_LLAMA_ID_HIDDEN_Q6_INDIRECT_OFFSET: usize = 16;

fn qwen_moe_llama_id_compact_blocks_enabled() -> bool {
    std::env::var("RNB_METAL_QWEN35_MOE_COMPACT_ID_BLOCKS")
        .ok()
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenMoeLlamaIdFallbackReason {
    PipelineUnavailable,
    UnsupportedQuant,
    ScratchOverBudget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenMoeLlamaIdError {
    InvalidShape,
    InvalidExpertArena,
    ExpertArenaMismatch,
    ExpertIdOutOfRange,
    TokenIdOutOfRange,
    InvalidSelectedExpertLayout,
    DispatchGridOverflow,
    AllocationFailed,
    DeviceMapFailed,
    CommandBufferFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct QwenMoeLlamaIdPlan {
    pub(crate) n_tokens: usize,
    pub(crate) n_expert: usize,
    pub(crate) n_expert_used: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) ffn_dim: usize,
    pub(crate) gate_bytes: usize,
    pub(crate) up_bytes: usize,
    pub(crate) down_rank_bytes: usize,
    pub(crate) selected_expert_bytes: usize,
    pub(crate) route_weight_bytes: usize,
    pub(crate) shared_route_weight_bytes: usize,
    pub(crate) tpe_bytes: usize,
    pub(crate) ids_bytes: usize,
    pub(crate) block_meta_bytes: usize,
    pub(crate) q8_bytes: usize,
    pub(crate) sparse_moe_out_bytes: usize,
    pub(crate) shared_out_bytes: usize,
    pub(crate) total_bytes: usize,
}

struct QwenMoePrefillRouteScratch {
    normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    router_logits_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    token_ids_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct QwenMoeLlamaIdCarrier {
    n_tokens: usize,
    n_expert: usize,
    n_expert_used: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    selected_experts_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    route_weights_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    shared_route_weights_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    tpe_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ids_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_meta_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    block_local_offset: usize,
    block_indirect_offset: usize,
    q8_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_rank_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    sparse_moe_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    shared_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    prefill_route: Option<QwenMoePrefillRouteScratch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenMoeLlamaIdPreflight {
    Run(QwenMoeLlamaIdPlan),
    Fallback(QwenMoeLlamaIdFallbackReason),
}

fn qwen_moe_llama_id_checked_aligned_bytes(
    factors: &[usize],
    element_bytes: usize,
) -> Result<usize, QwenMoeLlamaIdError> {
    let bytes = factors.iter().try_fold(element_bytes, |bytes, &factor| {
        bytes
            .checked_mul(factor)
            .ok_or(QwenMoeLlamaIdError::InvalidShape)
    })?;
    bytes
        .checked_add(QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT - 1)
        .map(|padded| {
            padded / QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT * QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT
        })
        .ok_or(QwenMoeLlamaIdError::InvalidShape)
}

fn qwen_moe_llama_id_checked_total(
    allocation_bytes: &[usize],
) -> Result<usize, QwenMoeLlamaIdError> {
    allocation_bytes.iter().try_fold(0usize, |total, &bytes| {
        total
            .checked_add(bytes)
            .ok_or(QwenMoeLlamaIdError::InvalidShape)
    })
}

fn qwen_moe_llama_id_q8k_bytes(rows: usize, dim: usize) -> Result<usize, QwenMoeLlamaIdError> {
    let blocks_per_row = dim
        .checked_add(255)
        .map(|rounded| rounded / 256)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    qwen_moe_llama_id_checked_aligned_bytes(&[rows, blocks_per_row], QWEN_MOE_LLAMA_Q8K_BLOCK_BYTES)
}

fn qwen_moe_llama_id_block_meta_layout(
    n_tokens: usize,
    n_expert_used: usize,
) -> Result<(usize, usize, usize), QwenMoeLlamaIdError> {
    let slots = n_tokens
        .checked_mul(n_expert_used)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let block_experts_bytes = slots
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let block_local_offset = block_experts_bytes;
    let block_local_end = block_local_offset
        .checked_add(block_experts_bytes)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let block_indirect_offset = block_local_end
        .checked_add(15)
        .map(|bytes| bytes / 16 * 16)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let total_bytes = block_indirect_offset
        .checked_add(QWEN_MOE_LLAMA_ID_INDIRECT_ARGS_BYTES)
        .and_then(|bytes| bytes.checked_add(QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT - 1))
        .map(|bytes| {
            bytes / QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT * QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT
        })
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    Ok((total_bytes, block_local_offset, block_indirect_offset))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_llama_id_preflight(
    pipeline_available: bool,
    quant_supported: bool,
    n_tokens: usize,
    n_expert: usize,
    n_expert_used: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    scratch_budget_bytes: usize,
) -> Result<QwenMoeLlamaIdPreflight, QwenMoeLlamaIdError> {
    if n_tokens == 0
        || n_expert == 0
        || n_expert_used == 0
        || n_expert_used > n_expert
        || n_expert_used > 8
        || hidden_dim == 0
        || ffn_dim == 0
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let slots = n_tokens
        .checked_mul(n_expert_used)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let gate_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[slots, ffn_dim], 4)?;
    let up_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[slots, ffn_dim], 4)?;
    let down_rank_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[slots, hidden_dim], 4)?;
    let selected_expert_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[slots], 4)?;
    let route_weight_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[slots], 4)?;
    let shared_route_weight_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[n_tokens], 4)?;
    let tpe_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[n_expert], 4)?;
    let ids_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[n_expert, n_tokens], 4)?;
    let (block_meta_bytes, _, _) = qwen_moe_llama_id_block_meta_layout(n_tokens, n_expert_used)?;
    let q8_bytes = qwen_moe_llama_id_q8k_bytes(n_tokens, hidden_dim)?
        .max(qwen_moe_llama_id_q8k_bytes(slots, ffn_dim)?);
    let sparse_moe_out_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[n_tokens, hidden_dim], 4)?;
    let shared_out_bytes = qwen_moe_llama_id_checked_aligned_bytes(&[n_tokens, hidden_dim], 4)?;
    let total_bytes = qwen_moe_llama_id_checked_total(&[
        gate_bytes,
        up_bytes,
        down_rank_bytes,
        selected_expert_bytes,
        route_weight_bytes,
        shared_route_weight_bytes,
        tpe_bytes,
        ids_bytes,
        block_meta_bytes,
        q8_bytes,
        sparse_moe_out_bytes,
        shared_out_bytes,
    ])?;
    let plan = QwenMoeLlamaIdPlan {
        n_tokens,
        n_expert,
        n_expert_used,
        hidden_dim,
        ffn_dim,
        gate_bytes,
        up_bytes,
        down_rank_bytes,
        selected_expert_bytes,
        route_weight_bytes,
        shared_route_weight_bytes,
        tpe_bytes,
        ids_bytes,
        block_meta_bytes,
        q8_bytes,
        sparse_moe_out_bytes,
        shared_out_bytes,
        total_bytes,
    };

    if !pipeline_available {
        return Ok(QwenMoeLlamaIdPreflight::Fallback(
            QwenMoeLlamaIdFallbackReason::PipelineUnavailable,
        ));
    }
    if !quant_supported {
        return Ok(QwenMoeLlamaIdPreflight::Fallback(
            QwenMoeLlamaIdFallbackReason::UnsupportedQuant,
        ));
    }
    if total_bytes > scratch_budget_bytes {
        return Ok(QwenMoeLlamaIdPreflight::Fallback(
            QwenMoeLlamaIdFallbackReason::ScratchOverBudget,
        ));
    }
    Ok(QwenMoeLlamaIdPreflight::Run(plan))
}

fn qwen_moe_prefill_route_scratch_bytes(
    n_tokens: usize,
    n_expert: usize,
) -> Result<[usize; 2], QwenMoeLlamaIdError> {
    Ok([
        qwen_moe_llama_id_checked_aligned_bytes(&[n_tokens, n_expert], 4)?,
        qwen_moe_llama_id_checked_aligned_bytes(&[n_tokens], 4)?,
    ])
}

fn qwen_moe_prefill_device_route_plan(
    pipeline_available: bool,
    quant_supported: bool,
    n_tokens: usize,
    n_expert: usize,
    n_expert_used: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<QwenMoeLlamaIdPreflight, QwenMoeLlamaIdError> {
    let mut plan = match qwen_moe_llama_id_preflight(
        pipeline_available,
        quant_supported,
        n_tokens,
        n_expert,
        n_expert_used,
        hidden_dim,
        ffn_dim,
        usize::MAX,
    )? {
        QwenMoeLlamaIdPreflight::Run(plan) => plan,
        fallback => return Ok(fallback),
    };
    // Device-route carriers are the shared-Q4 chain carrier. Their down-rank
    // allocation is reused as the canonical shared path's contiguous F16
    // norm+activation arena before sparse down overwrites it.
    let shared_f16_arena_bytes = qwen_moe_llama_id_checked_aligned_bytes(
        &[
            n_tokens,
            hidden_dim
                .checked_add(ffn_dim)
                .ok_or(QwenMoeLlamaIdError::InvalidShape)?,
        ],
        std::mem::size_of::<u16>(),
    )?;
    if shared_f16_arena_bytes > plan.down_rank_bytes {
        plan.total_bytes = plan
            .total_bytes
            .checked_sub(plan.down_rank_bytes)
            .and_then(|total| total.checked_add(shared_f16_arena_bytes))
            .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
        plan.down_rank_bytes = shared_f16_arena_bytes;
    }
    // The exact short-token chain quantizes into Q8_K. At and above the
    // large-chain threshold, every sparse matmul consumes the F16 carrier
    // slots instead, so the device-route carrier has no Q8 allocation.
    if n_tokens >= QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD {
        plan.total_bytes = plan
            .total_bytes
            .checked_sub(plan.q8_bytes)
            .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
        plan.q8_bytes = 0;
    }
    Ok(QwenMoeLlamaIdPreflight::Run(plan))
}

fn qwen_moe_llama_id_checked_physical_bytes(bytes: usize) -> Result<usize, QwenMoeLlamaIdError> {
    bytes
        .checked_add(QWEN_MOE_LLAMA_ID_PHYSICAL_BUFFER_ALIGNMENT - 1)
        .map(|padded| {
            padded / QWEN_MOE_LLAMA_ID_PHYSICAL_BUFFER_ALIGNMENT
                * QWEN_MOE_LLAMA_ID_PHYSICAL_BUFFER_ALIGNMENT
        })
        .ok_or(QwenMoeLlamaIdError::InvalidShape)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_device_route_required_bytes(
    n_tokens: usize,
    n_expert: usize,
    n_expert_used: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<usize, QwenMoeLlamaIdError> {
    let plan = match qwen_moe_prefill_device_route_plan(
        true,
        true,
        n_tokens,
        n_expert,
        n_expert_used,
        hidden_dim,
        ffn_dim,
    )? {
        QwenMoeLlamaIdPreflight::Run(plan) => plan,
        QwenMoeLlamaIdPreflight::Fallback(_) => {
            return Err(QwenMoeLlamaIdError::InvalidShape);
        }
    };
    qwen_moe_llama_carrier_allocation_bytes(plan)
        .into_iter()
        .chain(qwen_moe_prefill_route_scratch_bytes(n_tokens, n_expert)?)
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(qwen_moe_llama_id_checked_physical_bytes(bytes)?)
                .ok_or(QwenMoeLlamaIdError::InvalidShape)
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_device_route_preflight(
    pipeline_available: bool,
    quant_supported: bool,
    n_tokens: usize,
    n_expert: usize,
    n_expert_used: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    scratch_budget_bytes: usize,
) -> Result<QwenMoeLlamaIdPreflight, QwenMoeLlamaIdError> {
    if n_tokens >= (1usize << 31)
        || u32::try_from(n_expert).is_err()
        || u32::try_from(n_expert_used).is_err()
        || u32::try_from(hidden_dim).is_err()
        || u32::try_from(ffn_dim).is_err()
        || n_tokens
            .checked_mul(n_expert_used)
            .is_none_or(|slots| slots > i32::MAX as usize)
    {
        return Err(QwenMoeLlamaIdError::DispatchGridOverflow);
    }
    let plan = match qwen_moe_prefill_device_route_plan(
        pipeline_available,
        quant_supported,
        n_tokens,
        n_expert,
        n_expert_used,
        hidden_dim,
        ffn_dim,
    )? {
        QwenMoeLlamaIdPreflight::Run(plan) => plan,
        fallback => return Ok(fallback),
    };
    let total_bytes = qwen_moe_prefill_device_route_required_bytes(
        n_tokens,
        n_expert,
        n_expert_used,
        hidden_dim,
        ffn_dim,
    )?;
    if total_bytes > scratch_budget_bytes {
        return Ok(QwenMoeLlamaIdPreflight::Fallback(
            QwenMoeLlamaIdFallbackReason::ScratchOverBudget,
        ));
    }
    Ok(QwenMoeLlamaIdPreflight::Run(plan))
}

fn qwen_moe_llama_carrier_allocation_bytes(plan: QwenMoeLlamaIdPlan) -> [usize; 12] {
    [
        plan.selected_expert_bytes,
        plan.route_weight_bytes,
        plan.shared_route_weight_bytes,
        plan.tpe_bytes,
        plan.ids_bytes,
        plan.block_meta_bytes,
        plan.q8_bytes,
        plan.gate_bytes,
        plan.up_bytes,
        plan.down_rank_bytes,
        plan.sparse_moe_out_bytes,
        plan.shared_out_bytes,
    ]
}

fn qwen_moe_llama_validate_routes(
    n_tokens: usize,
    n_expert: usize,
    n_expert_used: usize,
    selected_experts: &[u32],
    route_weights: &[f32],
    shared_route_weights: &[f32],
) -> Result<(), QwenMoeLlamaIdError> {
    let slots = n_tokens
        .checked_mul(n_expert_used)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if selected_experts.len() != slots
        || route_weights.len() != slots
        || shared_route_weights.len() != n_tokens
    {
        return Err(QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
    }
    for token_routes in selected_experts.chunks_exact(n_expert_used) {
        for (rank, &expert) in token_routes.iter().enumerate() {
            if expert as usize >= n_expert {
                return Err(QwenMoeLlamaIdError::ExpertIdOutOfRange);
            }
            if token_routes[..rank].contains(&expert) {
                return Err(QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
            }
        }
    }
    Ok(())
}

fn qwen_moe_llama_validate_carrier_inputs(
    plan: QwenMoeLlamaIdPlan,
    selected_experts: &[u32],
    route_weights: &[f32],
    shared_route_weights: &[f32],
) -> Result<(), QwenMoeLlamaIdError> {
    if plan.n_expert_used == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let expected = match qwen_moe_llama_id_preflight(
        true,
        true,
        plan.n_tokens,
        plan.n_expert,
        plan.n_expert_used,
        plan.hidden_dim,
        plan.ffn_dim,
        usize::MAX,
    )? {
        QwenMoeLlamaIdPreflight::Run(expected) => expected,
        QwenMoeLlamaIdPreflight::Fallback(_) => return Err(QwenMoeLlamaIdError::InvalidShape),
    };
    if plan != expected {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    qwen_moe_llama_validate_routes(
        plan.n_tokens,
        plan.n_expert,
        plan.n_expert_used,
        selected_experts,
        route_weights,
        shared_route_weights,
    )
}

fn qwen_moe_llama_carrier_allocate_with<T>(
    plan: QwenMoeLlamaIdPlan,
    selected_experts: &[u32],
    route_weights: &[f32],
    shared_route_weights: &[f32],
    mut allocate: impl FnMut(usize) -> Option<T>,
) -> Result<Vec<T>, QwenMoeLlamaIdError> {
    qwen_moe_llama_validate_carrier_inputs(
        plan,
        selected_experts,
        route_weights,
        shared_route_weights,
    )?;
    let mut allocations = Vec::with_capacity(12);
    for bytes in qwen_moe_llama_carrier_allocation_bytes(plan) {
        allocations.push(allocate(bytes).ok_or(QwenMoeLlamaIdError::AllocationFailed)?);
    }
    Ok(allocations)
}

impl QwenMoeLlamaIdCarrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        plan: QwenMoeLlamaIdPlan,
        selected_experts: &[u32],
        route_weights: &[f32],
        shared_route_weights: &[f32],
    ) -> Result<Self, QwenMoeLlamaIdError> {
        let allocations = qwen_moe_llama_carrier_allocate_with(
            plan,
            selected_experts,
            route_weights,
            shared_route_weights,
            |bytes| {
                ctx.device
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            },
        )?;
        let [
            selected_experts_dev,
            route_weights_dev,
            shared_route_weights_dev,
            tpe_dev,
            ids_dev,
            block_meta_dev,
            q8_dev,
            gate_dev,
            up_dev,
            down_rank_dev,
            sparse_moe_out_dev,
            shared_out_dev,
        ]: [Retained<ProtocolObject<dyn MTLBuffer>>; 12] = allocations
            .try_into()
            .map_err(|_| QwenMoeLlamaIdError::AllocationFailed)?;

        let (_, block_local_offset, block_indirect_offset) =
            qwen_moe_llama_id_block_meta_layout(plan.n_tokens, plan.n_expert_used)?;
        let mut carrier = Self {
            n_tokens: plan.n_tokens,
            n_expert: plan.n_expert,
            n_expert_used: plan.n_expert_used,
            hidden_dim: plan.hidden_dim,
            ffn_dim: plan.ffn_dim,
            selected_experts_dev,
            route_weights_dev,
            shared_route_weights_dev,
            tpe_dev,
            ids_dev,
            block_meta_dev,
            block_local_offset,
            block_indirect_offset,
            q8_dev,
            gate_dev,
            up_dev,
            down_rank_dev,
            sparse_moe_out_dev,
            shared_out_dev,
            prefill_route: None,
        };
        carrier.write_routes(selected_experts, route_weights, shared_route_weights);
        Ok(carrier)
    }

    pub(crate) fn new_device_routes(
        ctx: &MetalContext,
        plan: QwenMoeLlamaIdPlan,
    ) -> Result<Self, QwenMoeLlamaIdError> {
        let expected = match qwen_moe_prefill_device_route_preflight(
            true,
            true,
            plan.n_tokens,
            plan.n_expert,
            plan.n_expert_used,
            plan.hidden_dim,
            plan.ffn_dim,
            usize::MAX,
        )? {
            QwenMoeLlamaIdPreflight::Run(expected) => expected,
            QwenMoeLlamaIdPreflight::Fallback(_) => {
                return Err(QwenMoeLlamaIdError::InvalidShape);
            }
        };
        if plan != expected {
            return Err(QwenMoeLlamaIdError::InvalidShape);
        }

        let mut allocations = qwen_moe_llama_carrier_allocation_bytes(plan)
            .into_iter()
            .filter(|&bytes| bytes != 0)
            .map(|bytes| {
                ctx.device
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(QwenMoeLlamaIdError::AllocationFailed)?;
        if plan.q8_bytes == 0 {
            // Q8 is unused by the large-token chain, but the encoder keeps a
            // valid carrier handle. Alias gate rather than creating a
            // zero-length Metal buffer.
            let q8_dev = allocations
                .get(6)
                .cloned()
                .ok_or(QwenMoeLlamaIdError::AllocationFailed)?;
            allocations.insert(6, q8_dev);
        }
        let [
            selected_experts_dev,
            route_weights_dev,
            shared_route_weights_dev,
            tpe_dev,
            ids_dev,
            block_meta_dev,
            q8_dev,
            gate_dev,
            up_dev,
            down_rank_dev,
            sparse_moe_out_dev,
            shared_out_dev,
        ]: [Retained<ProtocolObject<dyn MTLBuffer>>; 12] = allocations
            .try_into()
            .map_err(|_| QwenMoeLlamaIdError::AllocationFailed)?;

        let [router_logits_bytes, token_ids_bytes] =
            qwen_moe_prefill_route_scratch_bytes(plan.n_tokens, plan.n_expert)?;
        let normed_dev = sparse_moe_out_dev.clone();
        let router_logits_dev = ctx
            .device
            .newBufferWithLength_options(
                router_logits_bytes,
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or(QwenMoeLlamaIdError::AllocationFailed)?;
        let token_ids_dev = ctx
            .device
            .newBufferWithLength_options(token_ids_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(QwenMoeLlamaIdError::AllocationFailed)?;
        unsafe {
            let token_ids = token_ids_dev.contents().as_ptr() as *mut u32;
            for token in 0..plan.n_tokens {
                token_ids.add(token).write(token as u32);
            }
        }

        let (_, block_local_offset, block_indirect_offset) =
            qwen_moe_llama_id_block_meta_layout(plan.n_tokens, plan.n_expert_used)?;
        Ok(Self {
            n_tokens: plan.n_tokens,
            n_expert: plan.n_expert,
            n_expert_used: plan.n_expert_used,
            hidden_dim: plan.hidden_dim,
            ffn_dim: plan.ffn_dim,
            selected_experts_dev,
            route_weights_dev,
            shared_route_weights_dev,
            tpe_dev,
            ids_dev,
            block_meta_dev,
            block_local_offset,
            block_indirect_offset,
            q8_dev,
            gate_dev,
            up_dev,
            down_rank_dev,
            sparse_moe_out_dev,
            shared_out_dev,
            prefill_route: Some(QwenMoePrefillRouteScratch {
                normed_dev,
                router_logits_dev,
                token_ids_dev,
            }),
        })
    }

    pub(crate) fn refresh_routes(
        &mut self,
        selected_experts: &[u32],
        route_weights: &[f32],
        shared_route_weights: &[f32],
    ) -> Result<(), QwenMoeLlamaIdError> {
        qwen_moe_llama_validate_routes(
            self.n_tokens,
            self.n_expert,
            self.n_expert_used,
            selected_experts,
            route_weights,
            shared_route_weights,
        )?;
        self.write_routes(selected_experts, route_weights, shared_route_weights);
        Ok(())
    }

    fn write_routes(
        &mut self,
        selected_experts: &[u32],
        route_weights: &[f32],
        shared_route_weights: &[f32],
    ) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                selected_experts.as_ptr(),
                self.selected_experts_dev.contents().as_ptr() as *mut u32,
                selected_experts.len(),
            );
            std::ptr::copy_nonoverlapping(
                route_weights.as_ptr(),
                self.route_weights_dev.contents().as_ptr() as *mut f32,
                route_weights.len(),
            );
            std::ptr::copy_nonoverlapping(
                shared_route_weights.as_ptr(),
                self.shared_route_weights_dev.contents().as_ptr() as *mut f32,
                shared_route_weights.len(),
            );
        }
    }

    #[cfg(test)]
    pub(crate) fn debug_routes(&self) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let slots = self.n_tokens * self.n_expert_used;
        unsafe {
            (
                std::slice::from_raw_parts(
                    self.selected_experts_dev.contents().as_ptr() as *const u32,
                    slots,
                )
                .to_vec(),
                std::slice::from_raw_parts(
                    self.route_weights_dev.contents().as_ptr() as *const f32,
                    slots,
                )
                .to_vec(),
                std::slice::from_raw_parts(
                    self.shared_route_weights_dev.contents().as_ptr() as *const f32,
                    self.n_tokens,
                )
                .to_vec(),
            )
        }
    }

    pub(crate) fn output(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.sparse_moe_out_dev
    }

    pub(crate) fn sparse_output(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.sparse_moe_out_dev
    }

    pub(crate) fn shared_output(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.shared_out_dev
    }

    fn allocation_bytes(&self) -> usize {
        let base = [
            &*self.selected_experts_dev,
            &*self.route_weights_dev,
            &*self.shared_route_weights_dev,
            &*self.tpe_dev,
            &*self.ids_dev,
            &*self.block_meta_dev,
            &*self.q8_dev,
            &*self.gate_dev,
            &*self.up_dev,
            &*self.down_rank_dev,
            &*self.sparse_moe_out_dev,
            &*self.shared_out_dev,
        ];
        let physical_bytes = |bytes: usize| {
            if self.prefill_route.is_some() {
                bytes.div_ceil(QWEN_MOE_LLAMA_ID_PHYSICAL_BUFFER_ALIGNMENT)
                    * QWEN_MOE_LLAMA_ID_PHYSICAL_BUFFER_ALIGNMENT
            } else {
                bytes
            }
        };
        let mut total = 0;
        for index in 0..base.len() {
            let buffer = base[index];
            if base[..index]
                .iter()
                .any(|seen| std::ptr::eq::<ProtocolObject<dyn MTLBuffer>>(*seen, buffer))
            {
                continue;
            }
            total += physical_bytes(buffer.length());
        }
        if let Some(route) = self.prefill_route.as_ref() {
            total += physical_bytes(route.router_logits_dev.length());
            total += physical_bytes(route.token_ids_dev.length());
        }
        total
    }

    #[cfg(test)]
    fn prefill_route_norm_aliases_sparse_output(&self) -> bool {
        self.prefill_route.as_ref().is_some_and(|route| {
            std::ptr::eq::<ProtocolObject<dyn MTLBuffer>>(
                &*route.normed_dev,
                &*self.sparse_moe_out_dev,
            )
        })
    }

    #[cfg(test)]
    fn q8_aliases_gate(&self) -> bool {
        std::ptr::eq::<ProtocolObject<dyn MTLBuffer>>(&*self.q8_dev, &*self.gate_dev)
    }
}

pub(crate) fn qwen_moe_llama_map0_cpu(
    selected_experts: &[u32],
    n_tokens: usize,
    n_expert_used: usize,
    n_expert: usize,
) -> Result<(Vec<u32>, Vec<i32>), QwenMoeLlamaIdError> {
    if n_tokens == 0 || n_expert_used == 0 || n_expert == 0 || n_expert_used > n_expert {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let slots = n_tokens
        .checked_mul(n_expert_used)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if selected_experts.len() != slots {
        return Err(QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
    }
    let ids_len = n_expert
        .checked_mul(n_tokens)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let mut tpe = vec![0u32; n_expert];
    let mut ids = vec![i32::MIN; ids_len];

    for token in 0..n_tokens {
        let token_routes = &selected_experts[token * n_expert_used..(token + 1) * n_expert_used];
        for rank in 0..n_expert_used {
            let expert = token_routes[rank] as usize;
            if expert >= n_expert {
                return Err(QwenMoeLlamaIdError::ExpertIdOutOfRange);
            }
            if token_routes[..rank].contains(&token_routes[rank]) {
                return Err(QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
            }
            let route_slot = token
                .checked_mul(n_expert_used)
                .and_then(|base| base.checked_add(rank))
                .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
            let route_slot =
                i32::try_from(route_slot).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
            let local = tpe[expert] as usize;
            ids[expert * n_tokens + local] = route_slot;
            tpe[expert] = tpe[expert]
                .checked_add(1)
                .ok_or(QwenMoeLlamaIdError::DispatchGridOverflow)?;
        }
    }
    Ok((tpe, ids))
}

pub(crate) fn qwen_moe_llama_id_map0_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    selected_experts: &ProtocolObject<dyn MTLBuffer>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    n_tokens: usize,
    n_expert_used: usize,
    n_expert: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if n_tokens == 0 || n_expert_used == 0 || n_expert == 0 || n_expert_used > n_expert {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let slots = n_tokens
        .checked_mul(n_expert_used)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let ids_len = n_expert
        .checked_mul(n_tokens)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let selected_bytes = slots
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let tpe_bytes = n_expert
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let ids_bytes = ids_len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if selected_experts.length() < selected_bytes
        || tpe.length() < tpe_bytes
        || ids.length() < ids_bytes
    {
        return Err(QwenMoeLlamaIdError::InvalidSelectedExpertLayout);
    }
    let pipeline = ctx
        .qwen_moe_llama_id_map0_pipeline()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    if n_expert > pipeline.maxTotalThreadsPerThreadgroup() {
        return Err(QwenMoeLlamaIdError::DispatchGridOverflow);
    }
    let n_tokens =
        u32::try_from(n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_expert_used =
        u32::try_from(n_expert_used).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_expert =
        u32::try_from(n_expert).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(selected_experts), 0, 0);
        enc.setBuffer_offset_atIndex(Some(tpe), 0, 1);
        enc.setBuffer_offset_atIndex(Some(ids), 0, 2);
    }
    set_u32_bytes(enc, n_tokens, 3);
    set_u32_bytes(enc, n_expert_used, 4);
    set_u32_bytes(enc, n_expert, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: n_expert as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qwen_moe_llama_id_build_blocks_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    block_meta: &ProtocolObject<dyn MTLBuffer>,
    n_tokens: usize,
    n_expert_used: usize,
    n_expert: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if n_expert == 0 || hidden_dim == 0 || ffn_dim == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let (block_meta_bytes, block_local_offset, block_indirect_offset) =
        qwen_moe_llama_id_block_meta_layout(n_tokens, n_expert_used)?;
    let tpe_bytes = n_expert
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if tpe.length() < tpe_bytes || block_meta.length() < block_meta_bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let pipeline = ctx
        .qwen_moe_llama_id_build_blocks_pipeline()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    let n_expert =
        u32::try_from(n_expert).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let hidden_dim =
        u32::try_from(hidden_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let ffn_dim = u32::try_from(ffn_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(tpe), 0, 0);
        enc.setBuffer_offset_atIndex(Some(block_meta), 0, 1);
        enc.setBuffer_offset_atIndex(Some(block_meta), block_local_offset, 2);
        enc.setBuffer_offset_atIndex(Some(block_meta), block_indirect_offset, 3);
    }
    set_u32_bytes(enc, n_expert, 4);
    set_u32_bytes(enc, hidden_dim, 5);
    set_u32_bytes(enc, ffn_dim, 6);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn qwen_moe_llama_id_checked_bytes(
    factors: &[usize],
    element_bytes: usize,
) -> Result<usize, QwenMoeLlamaIdError> {
    factors.iter().try_fold(element_bytes, |bytes, &factor| {
        bytes
            .checked_mul(factor)
            .ok_or(QwenMoeLlamaIdError::InvalidShape)
    })
}

fn qwen_moe_llama_id_checked_tiles(size: usize, tile: usize) -> Result<usize, QwenMoeLlamaIdError> {
    size.checked_add(tile - 1)
        .map(|rounded| rounded / tile)
        .ok_or(QwenMoeLlamaIdError::DispatchGridOverflow)
}

#[allow(clippy::too_many_arguments)]
fn qwen_moe_llama_mul_mm_id_encode_inner(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    quant: QwenMoeLlamaIdQuant,
    input_type: QwenMoeLlamaIdInput,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    input: &ProtocolObject<dyn MTLBuffer>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    shape: QwenMoeLlamaIdMatmulShape,
    chain_small_only: bool,
) -> Result<(), QwenMoeLlamaIdError> {
    if shape.input_dim == 0
        || shape.input_dim % 256 != 0
        || shape.output_dim == 0
        || shape.n_tokens == 0
        || shape.n_expert == 0
        || shape.n_expert_used == 0
        || shape.n_expert_used > shape.n_expert
        || weight_offset % std::mem::align_of::<u16>() != 0
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    let block_bytes = match quant {
        QwenMoeLlamaIdQuant::Q4K => 144,
        QwenMoeLlamaIdQuant::Q5K => 176,
        QwenMoeLlamaIdQuant::Q6K => 210,
        QwenMoeLlamaIdQuant::Q8Zero => return Err(QwenMoeLlamaIdError::InvalidShape),
    };
    let expected_expert_bytes =
        qwen_moe_llama_id_checked_bytes(&[shape.output_dim, shape.input_dim / 256], block_bytes)?;
    if shape.expert_weight_bytes != expected_expert_bytes {
        return Err(QwenMoeLlamaIdError::ExpertArenaMismatch);
    }

    let input_element_bytes = match input_type {
        QwenMoeLlamaIdInput::F32 => std::mem::size_of::<f32>(),
        QwenMoeLlamaIdInput::F16 => std::mem::size_of::<u16>(),
    };
    let weight_bytes =
        qwen_moe_llama_id_checked_bytes(&[shape.n_expert, shape.expert_weight_bytes], 1)?;
    let weight_end = weight_offset
        .checked_add(weight_bytes)
        .ok_or(QwenMoeLlamaIdError::InvalidExpertArena)?;
    let input_bytes =
        qwen_moe_llama_id_checked_bytes(&[shape.n_tokens, shape.input_dim], input_element_bytes)?;
    let tpe_bytes = qwen_moe_llama_id_checked_bytes(&[shape.n_expert], std::mem::size_of::<u32>())?;
    let ids_bytes = qwen_moe_llama_id_checked_bytes(
        &[shape.n_expert, shape.n_tokens],
        std::mem::size_of::<i32>(),
    )?;
    let output_bytes = qwen_moe_llama_id_checked_bytes(
        &[shape.n_tokens, shape.n_expert_used, shape.output_dim],
        std::mem::size_of::<f32>(),
    )?;
    if weights.length() < weight_end {
        return Err(QwenMoeLlamaIdError::InvalidExpertArena);
    }
    if input.length() < input_bytes
        || tpe.length() < tpe_bytes
        || ids.length() < ids_bytes
        || output.length() < output_bytes
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    let pipeline = if chain_small_only {
        if input_type != QwenMoeLlamaIdInput::F32 {
            return Err(QwenMoeLlamaIdError::InvalidShape);
        }
        let variant = match quant {
            QwenMoeLlamaIdQuant::Q4K => 0,
            QwenMoeLlamaIdQuant::Q5K => 1,
            QwenMoeLlamaIdQuant::Q6K => 2,
            QwenMoeLlamaIdQuant::Q8Zero => unreachable!(),
        };
        ctx.qwen_moe_chain_small_pipeline(variant)
    } else {
        match (quant, input_type) {
            (QwenMoeLlamaIdQuant::Q4K, QwenMoeLlamaIdInput::F32) => {
                ctx.qwen_moe_llama_mul_mm_id_q4k_f32_pipeline()
            }
            (QwenMoeLlamaIdQuant::Q4K, QwenMoeLlamaIdInput::F16) => {
                ctx.qwen_moe_llama_mul_mm_id_q4k_f16_pipeline()
            }
            (QwenMoeLlamaIdQuant::Q5K, QwenMoeLlamaIdInput::F32) => {
                ctx.qwen_moe_llama_mul_mm_id_q5k_f32_pipeline()
            }
            (QwenMoeLlamaIdQuant::Q5K, QwenMoeLlamaIdInput::F16) => {
                ctx.qwen_moe_llama_mul_mm_id_q5k_f16_pipeline()
            }
            (QwenMoeLlamaIdQuant::Q6K, QwenMoeLlamaIdInput::F32) => {
                ctx.qwen_moe_llama_mul_mm_id_q6k_f32_pipeline()
            }
            (QwenMoeLlamaIdQuant::Q6K, QwenMoeLlamaIdInput::F16) => {
                ctx.qwen_moe_llama_mul_mm_id_q6k_f16_pipeline()
            }
            (QwenMoeLlamaIdQuant::Q8Zero, _) => unreachable!(),
        }
    }
    .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    if pipeline.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(QwenMoeLlamaIdError::DispatchGridOverflow);
    }

    let output_dim =
        u32::try_from(shape.output_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let input_dim =
        u32::try_from(shape.input_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_tokens =
        u32::try_from(shape.n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_expert_used = u32::try_from(shape.n_expert_used)
        .map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let expert_weight_bytes = u32::try_from(shape.expert_weight_bytes)
        .map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let token_tiles = qwen_moe_llama_id_checked_tiles(shape.n_tokens, 32)?;
    let row_tiles = qwen_moe_llama_id_checked_tiles(shape.output_dim, 64)?;

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(weights), weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(input), 0, 1);
        enc.setBuffer_offset_atIndex(Some(tpe), 0, 2);
        enc.setBuffer_offset_atIndex(Some(ids), 0, 3);
        enc.setBuffer_offset_atIndex(Some(output), 0, 4);
        enc.setThreadgroupMemoryLength_atIndex(8192, 0);
    }
    set_u32_bytes(enc, output_dim, 5);
    set_u32_bytes(enc, input_dim, 6);
    set_u32_bytes(enc, n_tokens, 7);
    set_u32_bytes(enc, n_expert_used, 8);
    set_u32_bytes(enc, expert_weight_bytes, 9);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: token_tiles,
            height: row_tiles,
            depth: shape.n_expert,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_llama_mul_mm_id_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    quant: QwenMoeLlamaIdQuant,
    input_type: QwenMoeLlamaIdInput,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    input: &ProtocolObject<dyn MTLBuffer>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    shape: QwenMoeLlamaIdMatmulShape,
) -> Result<(), QwenMoeLlamaIdError> {
    qwen_moe_llama_mul_mm_id_encode_inner(
        ctx,
        enc,
        quant,
        input_type,
        weights,
        weight_offset,
        input,
        tpe,
        ids,
        output,
        shape,
        false,
    )
}

pub(crate) fn qwen_moe_llama_quantize_q8k_f32_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n_elements: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if n_elements == 0 || n_elements % 256 != 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let input_bytes = qwen_moe_llama_id_checked_bytes(&[n_elements], std::mem::size_of::<f32>())?;
    let n_blocks = n_elements / 256;
    let output_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_blocks], QWEN_MOE_LLAMA_Q8K_BLOCK_BYTES)?;
    if input.length() < input_bytes || output.length() < output_bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let n_blocks_u32 =
        u32::try_from(n_blocks).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let pipeline = ctx
        .qwen_moe_llama_quantize_q8k_f32_pipeline()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
    }
    set_u32_bytes(enc, n_blocks_u32, 2);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: n_blocks,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: pipeline.threadExecutionWidth().max(1),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qwen_moe_llama_mul_mm_id_q8k_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    quant: QwenMoeLlamaIdQuant,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    input_q8k: &ProtocolObject<dyn MTLBuffer>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    shape: QwenMoeLlamaIdMatmulShape,
) -> Result<(), QwenMoeLlamaIdError> {
    if shape.input_dim == 0
        || shape.input_dim % 256 != 0
        || shape.output_dim == 0
        || shape.n_tokens == 0
        || shape.n_expert == 0
        || shape.n_expert_used == 0
        || shape.n_expert_used > shape.n_expert
        || weight_offset % std::mem::align_of::<u16>() != 0
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let block_bytes = match quant {
        QwenMoeLlamaIdQuant::Q4K => 144,
        QwenMoeLlamaIdQuant::Q6K => 210,
        QwenMoeLlamaIdQuant::Q5K | QwenMoeLlamaIdQuant::Q8Zero => {
            return Err(QwenMoeLlamaIdError::InvalidShape);
        }
    };
    let n_blocks = shape.input_dim / 256;
    let expected_expert_bytes =
        qwen_moe_llama_id_checked_bytes(&[shape.output_dim, n_blocks], block_bytes)?;
    if shape.expert_weight_bytes != expected_expert_bytes {
        return Err(QwenMoeLlamaIdError::ExpertArenaMismatch);
    }
    let weight_bytes =
        qwen_moe_llama_id_checked_bytes(&[shape.n_expert, shape.expert_weight_bytes], 1)?;
    let weight_end = weight_offset
        .checked_add(weight_bytes)
        .ok_or(QwenMoeLlamaIdError::InvalidExpertArena)?;
    let input_bytes = qwen_moe_llama_id_checked_bytes(
        &[shape.n_tokens, n_blocks],
        QWEN_MOE_LLAMA_Q8K_BLOCK_BYTES,
    )?;
    let tpe_bytes = qwen_moe_llama_id_checked_bytes(&[shape.n_expert], std::mem::size_of::<u32>())?;
    let ids_bytes = qwen_moe_llama_id_checked_bytes(
        &[shape.n_expert, shape.n_tokens],
        std::mem::size_of::<i32>(),
    )?;
    let output_bytes = qwen_moe_llama_id_checked_bytes(
        &[shape.n_tokens, shape.n_expert_used, shape.output_dim],
        std::mem::size_of::<f32>(),
    )?;
    if weights.length() < weight_end {
        return Err(QwenMoeLlamaIdError::InvalidExpertArena);
    }
    if input_q8k.length() < input_bytes
        || tpe.length() < tpe_bytes
        || ids.length() < ids_bytes
        || output.length() < output_bytes
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let pipeline = match quant {
        QwenMoeLlamaIdQuant::Q4K => ctx.qwen_moe_llama_mul_mm_id_q4k_q8k_pipeline(),
        QwenMoeLlamaIdQuant::Q6K => ctx.qwen_moe_llama_mul_mm_id_q6k_q8k_pipeline(),
        _ => unreachable!(),
    }
    .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(weights), weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(input_q8k), 0, 1);
        enc.setBuffer_offset_atIndex(Some(tpe), 0, 2);
        enc.setBuffer_offset_atIndex(Some(ids), 0, 3);
        enc.setBuffer_offset_atIndex(Some(output), 0, 4);
    }
    set_u32_bytes(
        enc,
        u32::try_from(shape.output_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        5,
    );
    set_u32_bytes(
        enc,
        u32::try_from(shape.input_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        6,
    );
    set_u32_bytes(
        enc,
        u32::try_from(shape.n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        7,
    );
    set_u32_bytes(
        enc,
        u32::try_from(shape.n_expert_used)
            .map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        8,
    );
    set_u32_bytes(
        enc,
        u32::try_from(shape.expert_weight_bytes)
            .map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        9,
    );
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: shape.output_dim,
            height: shape.n_tokens,
            depth: shape.n_expert,
        },
        MTLSize {
            width: 8,
            height: 4,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_chain_hybrid_mul_mm_id(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    quant: QwenMoeLlamaIdQuant,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    small_input_q8k: &ProtocolObject<dyn MTLBuffer>,
    large_input_f16: &ProtocolObject<dyn MTLBuffer>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    block_meta: &ProtocolObject<dyn MTLBuffer>,
    block_local_offset: usize,
    indirect_args_offset: usize,
    compact_blocks: bool,
    output: &ProtocolObject<dyn MTLBuffer>,
    shape: QwenMoeLlamaIdMatmulShape,
    route_slots: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    let large_variant = match (quant, compact_blocks) {
        (QwenMoeLlamaIdQuant::Q4K, true) => 1,
        (QwenMoeLlamaIdQuant::Q6K, true) => 3,
        (QwenMoeLlamaIdQuant::Q4K, false) => 4,
        (QwenMoeLlamaIdQuant::Q6K, false) => 5,
        (QwenMoeLlamaIdQuant::Q5K | QwenMoeLlamaIdQuant::Q8Zero, _) => {
            return Err(QwenMoeLlamaIdError::InvalidShape);
        }
    };
    let large_pipeline = ctx
        .qwen_moe_chain_large_pipeline(large_variant)
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    if large_pipeline.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(QwenMoeLlamaIdError::DispatchGridOverflow);
    }
    let large_input_bytes = qwen_moe_llama_id_checked_bytes(
        &[route_slots, shape.input_dim],
        std::mem::size_of::<u16>(),
    )?;
    let output_bytes = qwen_moe_llama_id_checked_bytes(
        &[route_slots, shape.output_dim],
        std::mem::size_of::<f32>(),
    )?;
    let block_bytes = qwen_moe_llama_id_checked_bytes(&[route_slots], std::mem::size_of::<u32>())?;
    let block_local_end = block_local_offset
        .checked_add(block_bytes)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let indirect_args_end = indirect_args_offset
        .checked_add(3 * std::mem::size_of::<u32>())
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if large_input_f16.length() < large_input_bytes
        || output.length() < output_bytes
        || block_meta.length() < block_bytes
        || block_meta.length() < block_local_end
        || block_meta.length() < indirect_args_end
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    if shape.n_tokens < QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD {
        qwen_moe_llama_mul_mm_id_q8k_encode(
            ctx,
            enc,
            quant,
            weights,
            weight_offset,
            small_input_q8k,
            tpe,
            ids,
            output,
            shape,
        )?;
    }

    let output_dim =
        u32::try_from(shape.output_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let input_dim =
        u32::try_from(shape.input_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_tokens =
        u32::try_from(shape.n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let top_k = u32::try_from(shape.n_expert_used)
        .map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let expert_weight_bytes = u32::try_from(shape.expert_weight_bytes)
        .map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;

    enc.setComputePipelineState(large_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(weights), weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(large_input_f16), 0, 1);
        enc.setBuffer_offset_atIndex(Some(tpe), 0, 2);
        enc.setBuffer_offset_atIndex(Some(ids), 0, 3);
        enc.setBuffer_offset_atIndex(Some(output), 0, 4);
        enc.setBuffer_offset_atIndex(Some(block_meta), 0, 10);
        enc.setBuffer_offset_atIndex(Some(block_meta), block_local_offset, 11);
        enc.setThreadgroupMemoryLength_atIndex(32_768, 0);
    }
    set_u32_bytes(enc, output_dim, 5);
    set_u32_bytes(enc, input_dim, 6);
    set_u32_bytes(enc, n_tokens, 7);
    set_u32_bytes(enc, top_k, 8);
    set_u32_bytes(enc, expert_weight_bytes, 9);
    if compact_blocks {
        unsafe {
            enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                block_meta,
                indirect_args_offset,
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }
    } else {
        let token_tiles =
            qwen_moe_llama_id_checked_tiles(shape.n_tokens, QWEN_MOE_LLAMA_ID_BLOCK_TILE)?;
        let row_tile = match quant {
            QwenMoeLlamaIdQuant::Q4K => 128,
            QwenMoeLlamaIdQuant::Q6K => 64,
            _ => unreachable!(),
        };
        let row_tiles = qwen_moe_llama_id_checked_tiles(shape.output_dim, row_tile)?;
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: row_tiles,
                height: token_tiles,
                depth: shape.n_expert,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

fn encode_qwen_moe_chain_cast_large_slots_f32_f16(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &ProtocolObject<dyn MTLBuffer>,
    tpe: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n_tokens: usize,
    n_expert: usize,
    route_slots: usize,
    dim: usize,
    input_by_token: bool,
    top_k: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if top_k == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let input_rows = if input_by_token {
        n_tokens
    } else {
        route_slots
    };
    let input_bytes =
        qwen_moe_llama_id_checked_bytes(&[input_rows, dim], std::mem::size_of::<f32>())?;
    let output_bytes =
        qwen_moe_llama_id_checked_bytes(&[route_slots, dim], std::mem::size_of::<u16>())?;
    let tpe_bytes = qwen_moe_llama_id_checked_bytes(&[n_expert], std::mem::size_of::<u32>())?;
    let ids_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_expert, n_tokens], std::mem::size_of::<i32>())?;
    if input.length() < input_bytes
        || output.length() < output_bytes
        || tpe.length() < tpe_bytes
        || ids.length() < ids_bytes
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let pipeline = ctx
        .qwen_moe_chain_cast_large_slots_pipeline()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    let n_tokens_u32 =
        u32::try_from(n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let dim_u32 = u32::try_from(dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let top_k_u32 = u32::try_from(top_k).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let thread_width = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    if thread_width == 0 {
        return Err(QwenMoeLlamaIdError::DispatchGridOverflow);
    }

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(tpe), 0, 1);
        enc.setBuffer_offset_atIndex(Some(ids), 0, 2);
        enc.setBuffer_offset_atIndex(Some(output), 0, 3);
    }
    set_u32_bytes(enc, n_tokens_u32, 4);
    set_u32_bytes(enc, dim_u32, 5);
    set_u32_bytes(enc, u32::from(input_by_token), 6);
    set_u32_bytes(enc, top_k_u32, 7);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: dim,
            height: n_tokens,
            depth: n_expert,
        },
        MTLSize {
            width: thread_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn qwen_moe_shared_mul_mm_q8_0_f32_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    input_dim: usize,
    output_dim: usize,
    n_tokens: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if input_dim == 0 || input_dim % 32 != 0 || output_dim == 0 || n_tokens == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    if weight_offset % std::mem::align_of::<u16>() != 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let weight_bytes = qwen_moe_llama_id_checked_bytes(&[output_dim, input_dim / 32], 34)?;
    let weight_end = weight_offset
        .checked_add(weight_bytes)
        .ok_or(QwenMoeLlamaIdError::InvalidExpertArena)?;
    let input_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, input_dim], std::mem::size_of::<f32>())?;
    let output_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, output_dim], std::mem::size_of::<f32>())?;
    if weights.length() < weight_end {
        return Err(QwenMoeLlamaIdError::InvalidExpertArena);
    }
    if input.length() < input_bytes || output.length() < output_bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    let pipeline = ctx
        .qwen_moe_llama_shared_mul_mm_q8_0_f32_pipeline()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    if pipeline.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(QwenMoeLlamaIdError::DispatchGridOverflow);
    }
    let output_dim =
        u32::try_from(output_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let input_dim =
        u32::try_from(input_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_tokens =
        u32::try_from(n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let token_tiles = qwen_moe_llama_id_checked_tiles(n_tokens as usize, 32)?;
    let row_tiles = qwen_moe_llama_id_checked_tiles(output_dim as usize, 64)?;

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(weights), weight_offset, 0);
        enc.setBuffer_offset_atIndex(Some(input), 0, 1);
        enc.setBuffer_offset_atIndex(Some(output), 0, 2);
        enc.setThreadgroupMemoryLength_atIndex(8192, 0);
    }
    set_u32_bytes(enc, output_dim, 3);
    set_u32_bytes(enc, input_dim, 4);
    set_u32_bytes(enc, n_tokens, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: token_tiles,
            height: row_tiles,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qwen_moe_shared_mul_mm_quant_f32_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    quant: QwenMoeLlamaIdQuant,
    weights: &ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    input_dim: usize,
    output_dim: usize,
    n_tokens: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    match quant {
        QwenMoeLlamaIdQuant::Q8Zero => qwen_moe_shared_mul_mm_q8_0_f32_encode(
            ctx,
            enc,
            weights,
            weight_offset,
            input,
            output,
            input_dim,
            output_dim,
            n_tokens,
        ),
        QwenMoeLlamaIdQuant::Q4K => crate::compute::encode_qwen_moe_shared_mul_mm_q4k_f32(
            ctx,
            enc,
            weights,
            weight_offset,
            input,
            output,
            output_dim,
            input_dim,
            n_tokens,
        ),
        QwenMoeLlamaIdQuant::Q6K => crate::compute::encode_qwen_moe_shared_mul_mm_q6k_f32(
            ctx,
            enc,
            weights,
            weight_offset,
            input,
            output,
            output_dim,
            input_dim,
            n_tokens,
        ),
        QwenMoeLlamaIdQuant::Q5K => Err(QwenMoeLlamaIdError::InvalidShape),
    }
}

fn qwen_moe_llama_expert_weight_bytes(
    quant: QwenMoeLlamaIdQuant,
    input_dim: usize,
    output_dim: usize,
) -> Result<usize, QwenMoeLlamaIdError> {
    if input_dim == 0 || input_dim % 256 != 0 || output_dim == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let block_bytes = match quant {
        QwenMoeLlamaIdQuant::Q4K => 144,
        QwenMoeLlamaIdQuant::Q5K => 176,
        QwenMoeLlamaIdQuant::Q6K => 210,
        QwenMoeLlamaIdQuant::Q8Zero => return Err(QwenMoeLlamaIdError::InvalidShape),
    };
    qwen_moe_llama_id_checked_bytes(&[output_dim, input_dim / 256], block_bytes)
}

fn qwen_moe_shared_expert_weight_bytes(
    quant: QwenMoeLlamaIdQuant,
    input_dim: usize,
    output_dim: usize,
) -> Result<usize, QwenMoeLlamaIdError> {
    if quant != QwenMoeLlamaIdQuant::Q8Zero {
        return qwen_moe_llama_expert_weight_bytes(quant, input_dim, output_dim);
    }
    if input_dim == 0 || input_dim % 32 != 0 || output_dim == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    qwen_moe_llama_id_checked_bytes(&[output_dim, input_dim / 32], 34)
}

fn qwen_moe_llama_swiglu_f32_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n_elements: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if n_elements == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let bytes = qwen_moe_llama_id_checked_bytes(&[n_elements], std::mem::size_of::<f32>())?;
    if gate.length() < bytes || up.length() < bytes || output.length() < bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let n_elements_u32 =
        u32::try_from(n_elements).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let pipeline = ctx
        .qwen_moe_llama_swiglu_f32_pipeline()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    let tg_width = pipeline.threadExecutionWidth().max(1);

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up), 0, 1);
        enc.setBuffer_offset_atIndex(Some(output), 0, 2);
    }
    set_u32_bytes(enc, n_elements_u32, 3);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_elements.div_ceil(tg_width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qwen_moe_llama_weighted_rank_reduce_f32_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    slot_values: &ProtocolObject<dyn MTLBuffer>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n_tokens: usize,
    n_rank: usize,
    n_rows: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if n_tokens == 0 || n_rank == 0 || n_rows == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let slot_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, n_rank, n_rows], std::mem::size_of::<f32>())?;
    let weight_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, n_rank], std::mem::size_of::<f32>())?;
    let output_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, n_rows], std::mem::size_of::<f32>())?;
    if slot_values.length() < slot_bytes
        || weights.length() < weight_bytes
        || output.length() < output_bytes
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let n_tokens_u32 =
        u32::try_from(n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_rank_u32 =
        u32::try_from(n_rank).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_rows_u32 =
        u32::try_from(n_rows).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_elements = n_tokens
        .checked_mul(n_rows)
        .ok_or(QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let pipeline = ctx
        .qwen_moe_llama_weighted_rank_reduce_f32_pipeline()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    let tg_width = pipeline.threadExecutionWidth().max(1);

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(slot_values), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
        enc.setBuffer_offset_atIndex(Some(output), 0, 2);
    }
    set_u32_bytes(enc, n_tokens_u32, 3);
    set_u32_bytes(enc, n_rank_u32, 4);
    set_u32_bytes(enc, n_rows_u32, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_elements.div_ceil(tg_width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qwen_moe_llama_expert_order_reduce_f32_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    slot_values: &ProtocolObject<dyn MTLBuffer>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    selected_experts: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n_tokens: usize,
    n_rank: usize,
    n_rows: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    if n_tokens == 0 || n_rank == 0 || n_rank > 8 || n_rows == 0 {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let slot_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, n_rank, n_rows], std::mem::size_of::<f32>())?;
    let weight_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, n_rank], std::mem::size_of::<f32>())?;
    let selected_expert_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, n_rank], std::mem::size_of::<u32>())?;
    let output_bytes =
        qwen_moe_llama_id_checked_bytes(&[n_tokens, n_rows], std::mem::size_of::<f32>())?;
    if slot_values.length() < slot_bytes
        || weights.length() < weight_bytes
        || selected_experts.length() < selected_expert_bytes
        || output.length() < output_bytes
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let n_tokens_u32 =
        u32::try_from(n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_rank_u32 =
        u32::try_from(n_rank).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_rows_u32 =
        u32::try_from(n_rows).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_elements = n_tokens
        .checked_mul(n_rows)
        .ok_or(QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let pipeline = ctx
        .qwen_moe_llama_expert_order_reduce_f32_pipeline()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    let tg_width = pipeline.threadExecutionWidth().max(1);

    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(slot_values), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weights), 0, 1);
        enc.setBuffer_offset_atIndex(Some(selected_experts), 0, 2);
        enc.setBuffer_offset_atIndex(Some(output), 0, 3);
    }
    set_u32_bytes(enc, n_tokens_u32, 4);
    set_u32_bytes(enc, n_rank_u32, 5);
    set_u32_bytes(enc, n_rows_u32, 6);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_elements.div_ceil(tg_width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_shared_q4_canonical_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeLlamaIdCarrier,
    norm_all: &ProtocolObject<dyn MTLBuffer>,
    shared_gate: &ProtocolObject<dyn MTLBuffer>,
    shared_gate_offset: usize,
    shared_up: &ProtocolObject<dyn MTLBuffer>,
    shared_up_offset: usize,
    shared_down: &ProtocolObject<dyn MTLBuffer>,
    shared_down_offset: usize,
    shared_down_quant: QwenMoeLlamaIdQuant,
) -> Result<(), QwenMoeLlamaIdError> {
    let norm_elements = carrier
        .n_tokens
        .checked_mul(carrier.hidden_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let activation_elements = carrier
        .n_tokens
        .checked_mul(carrier.ffn_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let norm_f16_bytes =
        qwen_moe_llama_id_checked_bytes(&[norm_elements], std::mem::size_of::<u16>())?;
    let activation_f16_bytes =
        qwen_moe_llama_id_checked_bytes(&[activation_elements], std::mem::size_of::<u16>())?;
    let arena_bytes = norm_f16_bytes
        .checked_add(activation_f16_bytes)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if carrier.down_rank_dev.length() < arena_bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    crate::compute::encode_cast_f32_to_f16_offset(
        ctx,
        enc,
        norm_all,
        &carrier.down_rank_dev,
        0,
        norm_elements,
    )
    .map_err(|_| QwenMoeLlamaIdError::InvalidShape)?;
    crate::compute::chain_barrier(ctx, enc);
    crate::compute::encode_gemm_q4k_tensorops_v2_offset(
        ctx,
        enc,
        shared_gate,
        shared_gate_offset,
        &carrier.down_rank_dev,
        0,
        &carrier.gate_dev,
        0,
        carrier.ffn_dim,
        carrier.hidden_dim,
        carrier.n_tokens,
    )
    .map_err(|_| QwenMoeLlamaIdError::InvalidShape)?;
    crate::compute::encode_gemm_q4k_tensorops_v2_offset(
        ctx,
        enc,
        shared_up,
        shared_up_offset,
        &carrier.down_rank_dev,
        0,
        &carrier.up_dev,
        0,
        carrier.ffn_dim,
        carrier.hidden_dim,
        carrier.n_tokens,
    )
    .map_err(|_| QwenMoeLlamaIdError::InvalidShape)?;
    crate::compute::chain_barrier(ctx, enc);
    crate::compute::encode_silu_mul_to_f16_offset(
        ctx,
        enc,
        &carrier.gate_dev,
        0,
        &carrier.up_dev,
        0,
        &carrier.down_rank_dev,
        norm_f16_bytes,
        activation_elements,
    )
    .map_err(|_| QwenMoeLlamaIdError::InvalidShape)?;
    crate::compute::chain_barrier(ctx, enc);
    match shared_down_quant {
        QwenMoeLlamaIdQuant::Q4K => crate::compute::encode_gemm_q4k_tensorops_v2_offset(
            ctx,
            enc,
            shared_down,
            shared_down_offset,
            &carrier.down_rank_dev,
            norm_f16_bytes,
            &carrier.shared_out_dev,
            0,
            carrier.hidden_dim,
            carrier.ffn_dim,
            carrier.n_tokens,
        ),
        QwenMoeLlamaIdQuant::Q6K => crate::compute::encode_gemm_q6k_tensorops_v2_offset(
            ctx,
            enc,
            shared_down,
            shared_down_offset,
            &carrier.down_rank_dev,
            norm_f16_bytes,
            &carrier.shared_out_dev,
            0,
            carrier.hidden_dim,
            carrier.ffn_dim,
            carrier.n_tokens,
        ),
        _ => return Err(QwenMoeLlamaIdError::InvalidShape),
    }
    .map_err(|_| QwenMoeLlamaIdError::InvalidShape)
}

#[allow(clippy::too_many_arguments)]
fn qwen_moe_llama_prefill_encode_inner(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeLlamaIdCarrier,
    norm_all: &ProtocolObject<dyn MTLBuffer>,
    gate_weights: &ProtocolObject<dyn MTLBuffer>,
    gate_weights_offset: usize,
    up_weights: &ProtocolObject<dyn MTLBuffer>,
    up_weights_offset: usize,
    down_weights: &ProtocolObject<dyn MTLBuffer>,
    down_weights_offset: usize,
    sparse_quant: QwenMoeLlamaIdQuantSet,
    shared_gate: &ProtocolObject<dyn MTLBuffer>,
    shared_gate_offset: usize,
    shared_up: &ProtocolObject<dyn MTLBuffer>,
    shared_up_offset: usize,
    shared_down: &ProtocolObject<dyn MTLBuffer>,
    shared_down_offset: usize,
    shared_quant: QwenMoeLlamaIdQuantSet,
    raw_sparse_snapshot: Option<&ProtocolObject<dyn MTLBuffer>>,
    map_reuse: bool,
    mut stage_sampler: Option<&mut QwenMoeLlamaIdStageSampler>,
) -> Result<QwenMoeLlamaIdTrace, QwenMoeLlamaIdError> {
    if carrier.n_tokens == 0
        || carrier.n_expert == 0
        || carrier.n_expert_used == 0
        || carrier.n_expert_used > carrier.n_expert
        || carrier.hidden_dim == 0
        || carrier.ffn_dim == 0
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let sparse_supported = sparse_quant.gate == QwenMoeLlamaIdQuant::Q4K
        && sparse_quant.up == QwenMoeLlamaIdQuant::Q4K
        && matches!(
            sparse_quant.down,
            QwenMoeLlamaIdQuant::Q4K | QwenMoeLlamaIdQuant::Q5K | QwenMoeLlamaIdQuant::Q6K
        );
    let shared_q8 = shared_quant
        == (QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        });
    let shared_q4 = shared_quant.gate == QwenMoeLlamaIdQuant::Q4K
        && shared_quant.up == QwenMoeLlamaIdQuant::Q4K
        && matches!(
            shared_quant.down,
            QwenMoeLlamaIdQuant::Q4K | QwenMoeLlamaIdQuant::Q6K
        );
    let chain_sparse_exact = carrier.prefill_route.is_some()
        && sparse_quant.gate == QwenMoeLlamaIdQuant::Q4K
        && sparse_quant.up == QwenMoeLlamaIdQuant::Q4K
        && sparse_quant.down == QwenMoeLlamaIdQuant::Q6K;
    let compact_id_blocks = chain_sparse_exact && qwen_moe_llama_id_compact_blocks_enabled();
    if !sparse_supported || !(shared_q8 || shared_q4) {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let norm_bytes = qwen_moe_llama_id_checked_bytes(
        &[carrier.n_tokens, carrier.hidden_dim],
        std::mem::size_of::<f32>(),
    )?;
    if norm_all.length() < norm_bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    let gate_shape = QwenMoeLlamaIdMatmulShape {
        input_dim: carrier.hidden_dim,
        output_dim: carrier.ffn_dim,
        n_tokens: carrier.n_tokens,
        n_expert: carrier.n_expert,
        n_expert_used: carrier.n_expert_used,
        expert_weight_bytes: qwen_moe_llama_expert_weight_bytes(
            sparse_quant.gate,
            carrier.hidden_dim,
            carrier.ffn_dim,
        )?,
    };
    let up_shape = QwenMoeLlamaIdMatmulShape {
        expert_weight_bytes: qwen_moe_llama_expert_weight_bytes(
            sparse_quant.up,
            carrier.hidden_dim,
            carrier.ffn_dim,
        )?,
        ..gate_shape
    };
    let down_shape = QwenMoeLlamaIdMatmulShape {
        input_dim: carrier.ffn_dim,
        output_dim: carrier.hidden_dim,
        n_tokens: carrier.n_tokens,
        n_expert: carrier.n_expert,
        // Down consumes slot-major SwiGLU. TOP_K=1 makes the ID kernel index input by
        // route_slot while preserving the original expert-major map stride.
        n_expert_used: 1,
        expert_weight_bytes: qwen_moe_llama_expert_weight_bytes(
            sparse_quant.down,
            carrier.ffn_dim,
            carrier.hidden_dim,
        )?,
    };
    let mut trace = QwenMoeLlamaIdTrace {
        scratch_bytes: carrier.allocation_bytes(),
        ..QwenMoeLlamaIdTrace::default()
    };
    let sparse_route_slots = carrier
        .n_tokens
        .checked_mul(carrier.n_expert_used)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;

    let encode_map =
        |trace: &mut QwenMoeLlamaIdTrace,
         stage_sampler: &mut Option<&mut QwenMoeLlamaIdStageSampler>| {
            qwen_moe_llama_id_stage_begin(stage_sampler, enc, QwenMoeLlamaIdStage::Map);
            qwen_moe_llama_id_map0_encode(
                ctx,
                enc,
                &carrier.selected_experts_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                carrier.n_tokens,
                carrier.n_expert_used,
                carrier.n_expert,
            )?;
            if compact_id_blocks {
                crate::compute::chain_barrier(ctx, enc);
                qwen_moe_llama_id_build_blocks_encode(
                    ctx,
                    enc,
                    &carrier.tpe_dev,
                    &carrier.block_meta_dev,
                    carrier.n_tokens,
                    carrier.n_expert_used,
                    carrier.n_expert,
                    carrier.hidden_dim,
                    carrier.ffn_dim,
                )?;
                trace.map_dispatches += 1;
            }
            qwen_moe_llama_id_stage_end(stage_sampler, enc);
            trace.map_dispatches += 1;
            Ok::<(), QwenMoeLlamaIdError>(())
        };

    if chain_sparse_exact && carrier.n_tokens < QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD {
        let norm_elements = carrier
            .n_tokens
            .checked_mul(carrier.hidden_dim)
            .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
        qwen_moe_llama_quantize_q8k_f32_encode(ctx, enc, norm_all, &carrier.q8_dev, norm_elements)?;
        crate::compute::chain_barrier(ctx, enc);
    }

    encode_map(&mut trace, &mut stage_sampler)?;
    crate::compute::chain_barrier(ctx, enc);
    if chain_sparse_exact {
        encode_qwen_moe_chain_cast_large_slots_f32_f16(
            ctx,
            enc,
            norm_all,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.down_rank_dev,
            carrier.n_tokens,
            carrier.n_expert,
            sparse_route_slots,
            carrier.hidden_dim,
            true,
            carrier.n_expert_used,
        )?;
        crate::compute::chain_barrier(ctx, enc);
    }
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Gate);
    if chain_sparse_exact {
        encode_qwen_moe_chain_hybrid_mul_mm_id(
            ctx,
            enc,
            sparse_quant.gate,
            gate_weights,
            gate_weights_offset,
            &carrier.q8_dev,
            &carrier.down_rank_dev,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.block_meta_dev,
            carrier.block_local_offset,
            carrier.block_indirect_offset + QWEN_MOE_LLAMA_ID_FFN_Q4_INDIRECT_OFFSET,
            compact_id_blocks,
            &carrier.gate_dev,
            gate_shape,
            sparse_route_slots,
        )?;
    } else {
        qwen_moe_llama_mul_mm_id_encode(
            ctx,
            enc,
            sparse_quant.gate,
            QwenMoeLlamaIdInput::F32,
            gate_weights,
            gate_weights_offset,
            norm_all,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.gate_dev,
            gate_shape,
        )?;
    }
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.gate_dispatches += 1;

    if !map_reuse {
        crate::compute::chain_barrier(ctx, enc);
        encode_map(&mut trace, &mut stage_sampler)?;
        crate::compute::chain_barrier(ctx, enc);
    }
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Up);
    if chain_sparse_exact {
        encode_qwen_moe_chain_hybrid_mul_mm_id(
            ctx,
            enc,
            sparse_quant.up,
            up_weights,
            up_weights_offset,
            &carrier.q8_dev,
            &carrier.down_rank_dev,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.block_meta_dev,
            carrier.block_local_offset,
            carrier.block_indirect_offset + QWEN_MOE_LLAMA_ID_FFN_Q4_INDIRECT_OFFSET,
            compact_id_blocks,
            &carrier.up_dev,
            up_shape,
            sparse_route_slots,
        )?;
    } else {
        qwen_moe_llama_mul_mm_id_encode(
            ctx,
            enc,
            sparse_quant.up,
            QwenMoeLlamaIdInput::F32,
            up_weights,
            up_weights_offset,
            norm_all,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.up_dev,
            up_shape,
        )?;
    }
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.up_dispatches += 1;
    crate::compute::chain_barrier(ctx, enc);

    let sparse_activation_elements = sparse_route_slots
        .checked_mul(carrier.ffn_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Activation);
    qwen_moe_llama_swiglu_f32_encode(
        ctx,
        enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.gate_dev,
        sparse_activation_elements,
    )?;
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.activation_dispatches += 1;

    if !map_reuse {
        encode_map(&mut trace, &mut stage_sampler)?;
    }
    crate::compute::chain_barrier(ctx, enc);
    if chain_sparse_exact {
        if carrier.n_tokens < QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD {
            qwen_moe_llama_quantize_q8k_f32_encode(
                ctx,
                enc,
                &carrier.gate_dev,
                &carrier.q8_dev,
                sparse_activation_elements,
            )?;
        }
        encode_qwen_moe_chain_cast_large_slots_f32_f16(
            ctx,
            enc,
            &carrier.gate_dev,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.up_dev,
            carrier.n_tokens,
            carrier.n_expert,
            sparse_route_slots,
            carrier.ffn_dim,
            false,
            1,
        )?;
        crate::compute::chain_barrier(ctx, enc);
    }
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Down);
    if chain_sparse_exact {
        encode_qwen_moe_chain_hybrid_mul_mm_id(
            ctx,
            enc,
            sparse_quant.down,
            down_weights,
            down_weights_offset,
            &carrier.q8_dev,
            &carrier.up_dev,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.block_meta_dev,
            carrier.block_local_offset,
            carrier.block_indirect_offset + QWEN_MOE_LLAMA_ID_HIDDEN_Q6_INDIRECT_OFFSET,
            compact_id_blocks,
            &carrier.down_rank_dev,
            down_shape,
            sparse_route_slots,
        )?;
    } else {
        qwen_moe_llama_mul_mm_id_encode(
            ctx,
            enc,
            sparse_quant.down,
            QwenMoeLlamaIdInput::F32,
            down_weights,
            down_weights_offset,
            &carrier.gate_dev,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.down_rank_dev,
            down_shape,
        )?;
    }
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.down_dispatches += 1;
    crate::compute::chain_barrier(ctx, enc);

    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Gate);
    qwen_moe_shared_mul_mm_quant_f32_encode(
        ctx,
        enc,
        shared_quant.gate,
        shared_gate,
        shared_gate_offset,
        norm_all,
        &carrier.gate_dev,
        carrier.hidden_dim,
        carrier.ffn_dim,
        carrier.n_tokens,
    )?;
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.gate_dispatches += 1;
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Up);
    qwen_moe_shared_mul_mm_quant_f32_encode(
        ctx,
        enc,
        shared_quant.up,
        shared_up,
        shared_up_offset,
        norm_all,
        &carrier.up_dev,
        carrier.hidden_dim,
        carrier.ffn_dim,
        carrier.n_tokens,
    )?;
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.up_dispatches += 1;
    crate::compute::chain_barrier(ctx, enc);
    let shared_activation_elements = carrier
        .n_tokens
        .checked_mul(carrier.ffn_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Activation);
    qwen_moe_llama_swiglu_f32_encode(
        ctx,
        enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.gate_dev,
        shared_activation_elements,
    )?;
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.activation_dispatches += 1;
    crate::compute::chain_barrier(ctx, enc);
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Down);
    qwen_moe_shared_mul_mm_quant_f32_encode(
        ctx,
        enc,
        shared_quant.down,
        shared_down,
        shared_down_offset,
        &carrier.gate_dev,
        &carrier.shared_out_dev,
        carrier.ffn_dim,
        carrier.hidden_dim,
        carrier.n_tokens,
    )?;
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.down_dispatches += 1;
    crate::compute::chain_barrier(ctx, enc);
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Reduce);
    qwen_moe_llama_weighted_rank_reduce_f32_encode(
        ctx,
        enc,
        &carrier.shared_out_dev,
        &carrier.shared_route_weights_dev,
        &carrier.shared_out_dev,
        carrier.n_tokens,
        1,
        carrier.hidden_dim,
    )?;
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.reduce_dispatches += 1;
    crate::compute::chain_barrier(ctx, enc);
    // All norm readers have completed before this point. The sparse rank
    // reduction overwrites every output element, so a device-route carrier may
    // now end the norm_all/sparse_moe_out alias lifetime.
    qwen_moe_llama_id_stage_begin(&mut stage_sampler, enc, QwenMoeLlamaIdStage::Reduce);
    qwen_moe_llama_expert_order_reduce_f32_encode(
        ctx,
        enc,
        &carrier.down_rank_dev,
        &carrier.route_weights_dev,
        &carrier.selected_experts_dev,
        &carrier.sparse_moe_out_dev,
        carrier.n_tokens,
        carrier.n_expert_used,
        carrier.hidden_dim,
    )?;
    qwen_moe_llama_id_stage_end(&mut stage_sampler, enc);
    trace.reduce_dispatches += 1;
    crate::compute::chain_barrier(ctx, enc);
    if let Some(snapshot) = raw_sparse_snapshot {
        let route = carrier
            .prefill_route
            .as_ref()
            .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
        encode_qwen_moe_prefill_hidden_copy(
            ctx,
            enc,
            route,
            &carrier.sparse_moe_out_dev,
            snapshot,
            carrier.n_tokens,
            carrier.hidden_dim,
        )?;
        crate::compute::chain_barrier(ctx, enc);
    }
    let output_elements = carrier
        .n_tokens
        .checked_mul(carrier.hidden_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    encode_residual_add_inline_dim(
        ctx,
        enc,
        &carrier.sparse_moe_out_dev,
        &carrier.shared_out_dev,
        output_elements,
    )?;
    Ok(trace)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_llama_prefill_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeLlamaIdCarrier,
    norm_all: &ProtocolObject<dyn MTLBuffer>,
    gate_weights: &ProtocolObject<dyn MTLBuffer>,
    gate_weights_offset: usize,
    up_weights: &ProtocolObject<dyn MTLBuffer>,
    up_weights_offset: usize,
    down_weights: &ProtocolObject<dyn MTLBuffer>,
    down_weights_offset: usize,
    sparse_quant: QwenMoeLlamaIdQuantSet,
    shared_gate: &ProtocolObject<dyn MTLBuffer>,
    shared_gate_offset: usize,
    shared_up: &ProtocolObject<dyn MTLBuffer>,
    shared_up_offset: usize,
    shared_down: &ProtocolObject<dyn MTLBuffer>,
    shared_down_offset: usize,
    shared_quant: QwenMoeLlamaIdQuantSet,
    map_reuse: bool,
    stage_sampler: Option<&mut QwenMoeLlamaIdStageSampler>,
) -> Result<QwenMoeLlamaIdTrace, QwenMoeLlamaIdError> {
    qwen_moe_llama_prefill_encode_inner(
        ctx,
        enc,
        carrier,
        norm_all,
        gate_weights,
        gate_weights_offset,
        up_weights,
        up_weights_offset,
        down_weights,
        down_weights_offset,
        sparse_quant,
        shared_gate,
        shared_gate_offset,
        shared_up,
        shared_up_offset,
        shared_down,
        shared_down_offset,
        shared_quant,
        None,
        map_reuse,
        stage_sampler,
    )
}

const QWEN_MOE_PREFILL_SHARED_SIGMOID_MODE: u32 = 0x8000_0000;

fn qwen_moe_prefill_shared_f32_buffer(
    ctx: &MetalContext,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, QwenMoeLlamaIdError> {
    if data.is_empty() {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
        .ok_or(QwenMoeLlamaIdError::AllocationFailed)?;
    unsafe {
        ctx.device
            .newBufferWithBytes_length_options(
                ptr,
                std::mem::size_of_val(data),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or(QwenMoeLlamaIdError::AllocationFailed)
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_prefill_f32_projection(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    n_out: usize,
    hidden_dim: usize,
    n_tokens: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    let n_out = u32::try_from(n_out).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let hidden_dim =
        u32::try_from(hidden_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    let n_tokens =
        u32::try_from(n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    enc.setComputePipelineState(&ctx.qwen_moe_prefill_router_f32_exact_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(weight), 0, 0);
        enc.setBuffer_offset_atIndex(Some(input), 0, 1);
        enc.setBuffer_offset_atIndex(Some(output), 0, 2);
    }
    set_u32_bytes(enc, n_out, 3);
    set_u32_bytes(enc, hidden_dim, 4);
    set_u32_bytes(enc, 0, 5);
    set_u32_bytes(enc, n_tokens, 6);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: (n_tokens as usize).div_ceil(16),
            height: (n_out as usize).div_ceil(16),
            depth: 1,
        },
        MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_prefill_rms_norm_exact(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), QwenMoeLlamaIdError> {
    let cols = u32::try_from(cols).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?;
    enc.setComputePipelineState(&ctx.qwen_moe_prefill_rms_norm_exact_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight), 0, 1);
        enc.setBuffer_offset_atIndex(Some(output), 0, 2);
    }
    set_u32_bytes(enc, cols, 3);
    set_f32_bytes(enc, eps, 4);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    crate::compute::chain_barrier(ctx, enc);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_prefill_route_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    hidden_residual: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w: &ProtocolObject<dyn MTLBuffer>,
    router_w: &ProtocolObject<dyn MTLBuffer>,
    shared_input_scale: &ProtocolObject<dyn MTLBuffer>,
    norm_eps: f32,
    carrier: &QwenMoeLlamaIdCarrier,
    policy: QwenRouteAlgorithm,
) -> Result<(), QwenMoeLlamaIdError> {
    match policy {
        QwenRouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1 => {}
    }
    if carrier.n_expert_used == 0
        || carrier.n_expert_used > carrier.n_expert
        || carrier.n_expert_used > 32
        || carrier.n_tokens >= (1usize << 31)
        || !norm_eps.is_finite()
        || norm_eps <= 0.0
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let hidden_bytes = qwen_moe_llama_id_checked_bytes(&[carrier.n_tokens, carrier.hidden_dim], 4)?;
    let norm_weight_bytes = qwen_moe_llama_id_checked_bytes(&[carrier.hidden_dim], 4)?;
    let router_weight_bytes =
        qwen_moe_llama_id_checked_bytes(&[carrier.n_expert, carrier.hidden_dim], 4)?;
    if hidden_residual.length() < hidden_bytes
        || ffn_norm_w.length() < norm_weight_bytes
        || router_w.length() < router_weight_bytes
        || shared_input_scale.length() < norm_weight_bytes
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let route = carrier
        .prefill_route
        .as_ref()
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;

    encode_qwen_prefill_rms_norm_exact(
        ctx,
        enc,
        hidden_residual,
        ffn_norm_w,
        &route.normed_dev,
        carrier.n_tokens,
        carrier.hidden_dim,
        norm_eps,
    )?;

    encode_qwen_moe_prefill_f32_projection(
        ctx,
        enc,
        router_w,
        &route.normed_dev,
        &route.router_logits_dev,
        carrier.n_expert,
        carrier.hidden_dim,
        carrier.n_tokens,
    )?;
    enc.setComputePipelineState(&ctx.qwen_moe_prefill_shared_gate_chain_exact_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&route.normed_dev), 0, 0);
        enc.setBuffer_offset_atIndex(Some(shared_input_scale), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&carrier.shared_route_weights_dev), 0, 2);
    }
    set_u32_bytes(
        enc,
        u32::try_from(carrier.hidden_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        3,
    );
    set_u32_bytes(
        enc,
        u32::try_from(carrier.n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        4,
    );
    let shared_thread_width = ctx
        .qwen_moe_prefill_shared_gate_chain_exact_pipeline
        .threadExecutionWidth()
        .max(1);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: carrier.n_tokens.div_ceil(shared_thread_width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: shared_thread_width,
            height: 1,
            depth: 1,
        },
    );
    crate::compute::chain_barrier(ctx, enc);

    enc.setComputePipelineState(&ctx.qwen_moe_prefill_topk_chain_exact_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&route.router_logits_dev), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&carrier.selected_experts_dev), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&carrier.route_weights_dev), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&carrier.shared_route_weights_dev), 0, 3);
    }
    set_u32_bytes(
        enc,
        u32::try_from(carrier.n_expert).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        4,
    );
    set_u32_bytes(
        enc,
        u32::try_from(carrier.n_expert_used)
            .map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        5,
    );
    set_u32_bytes(
        enc,
        u32::try_from(carrier.n_tokens).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?
            | QWEN_MOE_PREFILL_SHARED_SIGMOID_MODE,
        6,
    );
    let thread_width = ctx
        .qwen_moe_prefill_topk_chain_exact_pipeline
        .threadExecutionWidth()
        .max(1);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: carrier.n_tokens.div_ceil(thread_width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_width,
            height: 1,
            depth: 1,
        },
    );
    crate::compute::chain_barrier(ctx, enc);
    Ok(())
}

fn validate_qwen_moe_prefill_layer(
    carrier: &QwenMoeLlamaIdCarrier,
    moe: QwenMoePrefillBackendSpecRef<'_>,
) -> Result<(), QwenMoeLlamaIdError> {
    if moe.n_expert != carrier.n_expert
        || moe.n_expert_used != carrier.n_expert_used
        || moe.hidden_dim != carrier.hidden_dim
        || moe.ffn_dim != carrier.ffn_dim
        || moe.n_expert_used == 0
        || moe.n_expert_used > moe.n_expert
        || moe.n_expert_used > 32
        || moe.ffn_norm_w.len() != moe.hidden_dim
        || moe.shared_input_scale.len() != moe.hidden_dim
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let router_elements = moe
        .n_expert
        .checked_mul(moe.hidden_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if moe.router_w.len() != router_elements {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    let gate_expert_bytes =
        qwen_moe_llama_expert_weight_bytes(moe.sparse_quant.gate, moe.hidden_dim, moe.ffn_dim)?;
    let up_expert_bytes =
        qwen_moe_llama_expert_weight_bytes(moe.sparse_quant.up, moe.hidden_dim, moe.ffn_dim)?;
    let down_expert_bytes =
        qwen_moe_llama_expert_weight_bytes(moe.sparse_quant.down, moe.ffn_dim, moe.hidden_dim)?;
    if moe.gate_expert_bytes != gate_expert_bytes
        || moe.up_expert_bytes != up_expert_bytes
        || moe.down_expert_bytes != down_expert_bytes
        || moe.gate_all.len()
            != gate_expert_bytes
                .checked_mul(moe.n_expert)
                .ok_or(QwenMoeLlamaIdError::InvalidShape)?
        || moe.up_all.len()
            != up_expert_bytes
                .checked_mul(moe.n_expert)
                .ok_or(QwenMoeLlamaIdError::InvalidShape)?
        || moe.down_all.len()
            != down_expert_bytes
                .checked_mul(moe.n_expert)
                .ok_or(QwenMoeLlamaIdError::InvalidShape)?
    {
        return Err(QwenMoeLlamaIdError::ExpertArenaMismatch);
    }

    let shared_gate_bytes =
        qwen_moe_shared_expert_weight_bytes(moe.shared_quant.gate, moe.hidden_dim, moe.ffn_dim)?;
    let shared_up_bytes =
        qwen_moe_shared_expert_weight_bytes(moe.shared_quant.up, moe.hidden_dim, moe.ffn_dim)?;
    let shared_down_bytes =
        qwen_moe_shared_expert_weight_bytes(moe.shared_quant.down, moe.ffn_dim, moe.hidden_dim)?;
    if moe.shared_gate.len() != shared_gate_bytes
        || moe.shared_up.len() != shared_up_bytes
        || moe.shared_down.len() != shared_down_bytes
    {
        return Err(QwenMoeLlamaIdError::ExpertArenaMismatch);
    }

    let sparse_supported = moe.sparse_quant.gate == QwenMoeLlamaIdQuant::Q4K
        && moe.sparse_quant.up == QwenMoeLlamaIdQuant::Q4K
        && matches!(
            moe.sparse_quant.down,
            QwenMoeLlamaIdQuant::Q4K | QwenMoeLlamaIdQuant::Q5K | QwenMoeLlamaIdQuant::Q6K
        );
    let shared_q8 = moe.shared_quant
        == (QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        });
    let shared_q4 = moe.shared_quant.gate == QwenMoeLlamaIdQuant::Q4K
        && moe.shared_quant.up == QwenMoeLlamaIdQuant::Q4K
        && matches!(
            moe.shared_quant.down,
            QwenMoeLlamaIdQuant::Q4K | QwenMoeLlamaIdQuant::Q6K
        );
    if !sparse_supported || !(shared_q8 || shared_q4) {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    Ok(())
}

fn encode_qwen_moe_prefill_hidden_copy(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    route: &QwenMoePrefillRouteScratch,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    n_tokens: usize,
    hidden_dim: usize,
) -> Result<(), QwenMoeLlamaIdError> {
    let total_elements = n_tokens
        .checked_mul(hidden_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    enc.setComputePipelineState(&ctx.qwen_moe_prefill_gather_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(hidden_in), 0, 0);
        enc.setBuffer_offset_atIndex(Some(hidden_out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&route.token_ids_dev), 0, 2);
    }
    set_u32_bytes(
        enc,
        u32::try_from(hidden_dim).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        3,
    );
    set_u32_bytes(enc, 0, 4);
    set_u32_bytes(
        enc,
        u32::try_from(total_elements).map_err(|_| QwenMoeLlamaIdError::DispatchGridOverflow)?,
        5,
    );
    let thread_width = ctx
        .qwen_moe_prefill_gather_pipeline
        .threadExecutionWidth()
        .max(1);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: total_elements.div_ceil(thread_width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_prefill_stage_snapshots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeLlamaIdCarrier,
    hidden_final: &ProtocolObject<dyn MTLBuffer>,
    sparse_dst: &ProtocolObject<dyn MTLBuffer>,
    shared_dst: &ProtocolObject<dyn MTLBuffer>,
    final_dst: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), QwenMoeLlamaIdError> {
    let route = carrier
        .prefill_route
        .as_ref()
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let elements = carrier
        .n_tokens
        .checked_mul(carrier.hidden_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if hidden_final.length() < bytes
        || sparse_dst.length() < bytes
        || shared_dst.length() < bytes
        || final_dst.length() < bytes
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }

    // sparse_output is sparse + shared at this point. The debug caller can recover
    // raw sparse as combined - shared after the command buffer completes.
    encode_qwen_moe_prefill_hidden_copy(
        ctx,
        enc,
        route,
        carrier.sparse_output(),
        sparse_dst,
        carrier.n_tokens,
        carrier.hidden_dim,
    )?;
    encode_qwen_moe_prefill_hidden_copy(
        ctx,
        enc,
        route,
        carrier.shared_output(),
        shared_dst,
        carrier.n_tokens,
        carrier.hidden_dim,
    )?;
    encode_qwen_moe_prefill_hidden_copy(
        ctx,
        enc,
        route,
        hidden_final,
        final_dst,
        carrier.n_tokens,
        carrier.hidden_dim,
    )?;
    crate::compute::chain_barrier(ctx, enc);
    Ok(())
}

fn encode_qwen_moe_prefill_layer_ops_inner(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeLlamaIdCarrier,
    moe: QwenMoePrefillBackendSpecRef<'_>,
    hidden_residual: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    raw_sparse_snapshot: Option<&ProtocolObject<dyn MTLBuffer>>,
    hidden_in_snapshot: Option<&ProtocolObject<dyn MTLBuffer>>,
    norm_snapshot: Option<&ProtocolObject<dyn MTLBuffer>>,
    tpe_snapshot: Option<&ProtocolObject<dyn MTLBuffer>>,
    ids_snapshot: Option<&ProtocolObject<dyn MTLBuffer>>,
) -> Result<QwenMoeLlamaIdTrace, QwenMoeLlamaIdError> {
    validate_qwen_moe_prefill_layer(carrier, moe)?;
    if std::ptr::eq(hidden_residual, hidden_out) {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let hidden_elements = carrier
        .n_tokens
        .checked_mul(carrier.hidden_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let hidden_bytes = hidden_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if hidden_residual.length() < hidden_bytes || hidden_out.length() < hidden_bytes {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let route = carrier
        .prefill_route
        .as_ref()
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    if let Some(snapshot) = hidden_in_snapshot {
        encode_qwen_moe_prefill_hidden_copy(
            ctx,
            enc,
            route,
            hidden_residual,
            snapshot,
            carrier.n_tokens,
            carrier.hidden_dim,
        )?;
    }

    let ffn_norm_w = qwen_moe_prefill_shared_f32_buffer(ctx, moe.ffn_norm_w)?;
    let router_w = qwen_moe_prefill_shared_f32_buffer(ctx, moe.router_w)?;
    let shared_input_scale = qwen_moe_prefill_shared_f32_buffer(ctx, moe.shared_input_scale)?;
    let (gate_all, gate_offset) = crate::compute::wrap_nocopy(ctx, moe.gate_all);
    let (up_all, up_offset) = crate::compute::wrap_nocopy(ctx, moe.up_all);
    let (down_all, down_offset) = crate::compute::wrap_nocopy(ctx, moe.down_all);
    let (shared_gate, shared_gate_offset) = crate::compute::wrap_nocopy(ctx, moe.shared_gate);
    let (shared_up, shared_up_offset) = crate::compute::wrap_nocopy(ctx, moe.shared_up);
    let (shared_down, shared_down_offset) = crate::compute::wrap_nocopy(ctx, moe.shared_down);

    encode_qwen_prefill_route_ops(
        ctx,
        enc,
        hidden_residual,
        &ffn_norm_w,
        &router_w,
        &shared_input_scale,
        moe.norm_eps,
        carrier,
        moe.route_algorithm,
    )?;
    if let Some(snapshot) = norm_snapshot {
        encode_qwen_moe_prefill_hidden_copy(
            ctx,
            enc,
            route,
            &route.normed_dev,
            snapshot,
            carrier.n_tokens,
            carrier.hidden_dim,
        )?;
    }
    let trace = qwen_moe_llama_prefill_encode_inner(
        ctx,
        enc,
        carrier,
        &route.normed_dev,
        &gate_all,
        gate_offset as usize,
        &up_all,
        up_offset as usize,
        &down_all,
        down_offset as usize,
        moe.sparse_quant,
        &shared_gate,
        shared_gate_offset as usize,
        &shared_up,
        shared_up_offset as usize,
        &shared_down,
        shared_down_offset as usize,
        moe.shared_quant,
        raw_sparse_snapshot,
        true,
        None,
    )?;
    if let Some(snapshot) = tpe_snapshot {
        encode_qwen_moe_prefill_hidden_copy(
            ctx,
            enc,
            route,
            &carrier.tpe_dev,
            snapshot,
            1,
            carrier.n_expert,
        )?;
    }
    if let Some(snapshot) = ids_snapshot {
        let ids_elements = carrier
            .n_expert
            .checked_mul(carrier.n_tokens)
            .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
        encode_qwen_moe_prefill_hidden_copy(
            ctx,
            enc,
            route,
            &carrier.ids_dev,
            snapshot,
            1,
            ids_elements,
        )?;
    }
    crate::compute::chain_barrier(ctx, enc);
    encode_qwen_moe_prefill_hidden_copy(
        ctx,
        enc,
        route,
        hidden_residual,
        hidden_out,
        carrier.n_tokens,
        carrier.hidden_dim,
    )?;
    crate::compute::chain_barrier(ctx, enc);
    encode_residual_add_inline_dim(ctx, enc, hidden_out, carrier.output(), hidden_elements)?;
    crate::compute::chain_barrier(ctx, enc);
    Ok(trace)
}

pub(crate) fn encode_qwen_moe_prefill_layer_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeLlamaIdCarrier,
    moe: QwenMoePrefillBackendSpecRef<'_>,
    hidden_residual: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
) -> Result<QwenMoeLlamaIdTrace, QwenMoeLlamaIdError> {
    encode_qwen_moe_prefill_layer_ops_inner(
        ctx,
        enc,
        carrier,
        moe,
        hidden_residual,
        hidden_out,
        None,
        None,
        None,
        None,
        None,
    )
}

pub(crate) fn encode_qwen_moe_prefill_layer_ops_with_stage_snapshots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeLlamaIdCarrier,
    moe: QwenMoePrefillBackendSpecRef<'_>,
    hidden_residual: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    raw_sparse_snapshot: &ProtocolObject<dyn MTLBuffer>,
    hidden_in_snapshot: &ProtocolObject<dyn MTLBuffer>,
    norm_snapshot: &ProtocolObject<dyn MTLBuffer>,
    tpe_snapshot: &ProtocolObject<dyn MTLBuffer>,
    ids_snapshot: &ProtocolObject<dyn MTLBuffer>,
) -> Result<QwenMoeLlamaIdTrace, QwenMoeLlamaIdError> {
    encode_qwen_moe_prefill_layer_ops_inner(
        ctx,
        enc,
        carrier,
        moe,
        hidden_residual,
        hidden_out,
        Some(raw_sparse_snapshot),
        Some(hidden_in_snapshot),
        Some(norm_snapshot),
        Some(tpe_snapshot),
        Some(ids_snapshot),
    )
}

fn qwen_moe_llama_id_trace_split_stage<F>(
    ctx: &MetalContext,
    encode: F,
) -> Result<f64, QwenMoeLlamaIdError>
where
    F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> Result<(), QwenMoeLlamaIdError>,
{
    let command = ctx
        .queue
        .commandBuffer()
        .ok_or(QwenMoeLlamaIdError::CommandBufferFailed)?;
    let encoder = crate::compute::chain_compute_encoder(ctx, &command);
    if let Err(error) = encode(&encoder) {
        encoder.endEncoding();
        return Err(error);
    }
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(QwenMoeLlamaIdError::CommandBufferFailed);
    }
    let gpu_start = command.GPUStartTime();
    let gpu_end = command.GPUEndTime();
    if !gpu_start.is_finite() || !gpu_end.is_finite() || gpu_end <= gpu_start {
        return Err(QwenMoeLlamaIdError::CommandBufferFailed);
    }
    Ok((gpu_end - gpu_start) * 1000.0)
}

fn qwen_moe_llama_id_trace_add_stage(
    trace: &mut QwenMoeLlamaIdTrace,
    stage: QwenMoeLlamaIdStage,
    gpu_ms: f64,
) {
    let slot = match stage {
        QwenMoeLlamaIdStage::Map => &mut trace.map_ms,
        QwenMoeLlamaIdStage::Gate => &mut trace.gate_ms,
        QwenMoeLlamaIdStage::Up => &mut trace.up_ms,
        QwenMoeLlamaIdStage::Activation => &mut trace.activation_ms,
        QwenMoeLlamaIdStage::Down => &mut trace.down_ms,
        QwenMoeLlamaIdStage::Reduce => &mut trace.reduce_ms,
    };
    *slot = Some(slot.unwrap_or(0.0) + gpu_ms);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_llama_prefill_trace_split(
    ctx: &MetalContext,
    carrier: &QwenMoeLlamaIdCarrier,
    norm_all: &ProtocolObject<dyn MTLBuffer>,
    gate_weights: &ProtocolObject<dyn MTLBuffer>,
    gate_weights_offset: usize,
    up_weights: &ProtocolObject<dyn MTLBuffer>,
    up_weights_offset: usize,
    down_weights: &ProtocolObject<dyn MTLBuffer>,
    down_weights_offset: usize,
    sparse_quant: QwenMoeLlamaIdQuantSet,
    shared_gate: &ProtocolObject<dyn MTLBuffer>,
    shared_gate_offset: usize,
    shared_up: &ProtocolObject<dyn MTLBuffer>,
    shared_up_offset: usize,
    shared_down: &ProtocolObject<dyn MTLBuffer>,
    shared_down_offset: usize,
    shared_quant: QwenMoeLlamaIdQuantSet,
    map_reuse: bool,
    counter_fallback_reason: &'static str,
) -> Result<(QwenMoeLlamaIdTrace, f64), QwenMoeLlamaIdError> {
    if carrier.n_tokens == 0
        || carrier.n_expert == 0
        || carrier.n_expert_used == 0
        || carrier.n_expert_used > carrier.n_expert
        || carrier.hidden_dim == 0
        || carrier.ffn_dim == 0
    {
        return Err(QwenMoeLlamaIdError::InvalidShape);
    }
    let gate_shape = QwenMoeLlamaIdMatmulShape {
        input_dim: carrier.hidden_dim,
        output_dim: carrier.ffn_dim,
        n_tokens: carrier.n_tokens,
        n_expert: carrier.n_expert,
        n_expert_used: carrier.n_expert_used,
        expert_weight_bytes: qwen_moe_llama_expert_weight_bytes(
            sparse_quant.gate,
            carrier.hidden_dim,
            carrier.ffn_dim,
        )?,
    };
    let up_shape = QwenMoeLlamaIdMatmulShape {
        expert_weight_bytes: qwen_moe_llama_expert_weight_bytes(
            sparse_quant.up,
            carrier.hidden_dim,
            carrier.ffn_dim,
        )?,
        ..gate_shape
    };
    let down_shape = QwenMoeLlamaIdMatmulShape {
        input_dim: carrier.ffn_dim,
        output_dim: carrier.hidden_dim,
        n_tokens: carrier.n_tokens,
        n_expert: carrier.n_expert,
        n_expert_used: 1,
        expert_weight_bytes: qwen_moe_llama_expert_weight_bytes(
            sparse_quant.down,
            carrier.ffn_dim,
            carrier.hidden_dim,
        )?,
    };
    let mut trace = QwenMoeLlamaIdTrace {
        scratch_bytes: carrier.allocation_bytes(),
        stage_timing_source: Some("trace_split_command_buffers"),
        stage_timing_fallback_reason: Some(counter_fallback_reason),
        ..QwenMoeLlamaIdTrace::default()
    };
    let mut total_gpu_ms = 0.0;

    let encode_map = |trace: &mut QwenMoeLlamaIdTrace, total_gpu_ms: &mut f64| {
        let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
            qwen_moe_llama_id_map0_encode(
                ctx,
                enc,
                &carrier.selected_experts_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                carrier.n_tokens,
                carrier.n_expert_used,
                carrier.n_expert,
            )
        })?;
        trace.map_dispatches += 1;
        qwen_moe_llama_id_trace_add_stage(trace, QwenMoeLlamaIdStage::Map, gpu_ms);
        *total_gpu_ms += gpu_ms;
        Ok::<(), QwenMoeLlamaIdError>(())
    };
    encode_map(&mut trace, &mut total_gpu_ms)?;

    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_llama_mul_mm_id_encode(
            ctx,
            enc,
            sparse_quant.gate,
            QwenMoeLlamaIdInput::F32,
            gate_weights,
            gate_weights_offset,
            norm_all,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.gate_dev,
            gate_shape,
        )
    })?;
    trace.gate_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Gate, gpu_ms);
    total_gpu_ms += gpu_ms;

    if !map_reuse {
        encode_map(&mut trace, &mut total_gpu_ms)?;
    }
    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_llama_mul_mm_id_encode(
            ctx,
            enc,
            sparse_quant.up,
            QwenMoeLlamaIdInput::F32,
            up_weights,
            up_weights_offset,
            norm_all,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.up_dev,
            up_shape,
        )
    })?;
    trace.up_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Up, gpu_ms);
    total_gpu_ms += gpu_ms;

    let sparse_activation_elements = carrier
        .n_tokens
        .checked_mul(carrier.n_expert_used)
        .and_then(|slots| slots.checked_mul(carrier.ffn_dim))
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_llama_swiglu_f32_encode(
            ctx,
            enc,
            &carrier.gate_dev,
            &carrier.up_dev,
            &carrier.gate_dev,
            sparse_activation_elements,
        )
    })?;
    trace.activation_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Activation, gpu_ms);
    total_gpu_ms += gpu_ms;

    if !map_reuse {
        encode_map(&mut trace, &mut total_gpu_ms)?;
    }
    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_llama_mul_mm_id_encode(
            ctx,
            enc,
            sparse_quant.down,
            QwenMoeLlamaIdInput::F32,
            down_weights,
            down_weights_offset,
            &carrier.gate_dev,
            &carrier.tpe_dev,
            &carrier.ids_dev,
            &carrier.down_rank_dev,
            down_shape,
        )
    })?;
    trace.down_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Down, gpu_ms);
    total_gpu_ms += gpu_ms;

    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_llama_expert_order_reduce_f32_encode(
            ctx,
            enc,
            &carrier.down_rank_dev,
            &carrier.route_weights_dev,
            &carrier.selected_experts_dev,
            &carrier.sparse_moe_out_dev,
            carrier.n_tokens,
            carrier.n_expert_used,
            carrier.hidden_dim,
        )
    })?;
    trace.reduce_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Reduce, gpu_ms);
    total_gpu_ms += gpu_ms;

    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_shared_mul_mm_quant_f32_encode(
            ctx,
            enc,
            shared_quant.gate,
            shared_gate,
            shared_gate_offset,
            norm_all,
            &carrier.gate_dev,
            carrier.hidden_dim,
            carrier.ffn_dim,
            carrier.n_tokens,
        )
    })?;
    trace.gate_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Gate, gpu_ms);
    total_gpu_ms += gpu_ms;

    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_shared_mul_mm_quant_f32_encode(
            ctx,
            enc,
            shared_quant.up,
            shared_up,
            shared_up_offset,
            norm_all,
            &carrier.up_dev,
            carrier.hidden_dim,
            carrier.ffn_dim,
            carrier.n_tokens,
        )
    })?;
    trace.up_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Up, gpu_ms);
    total_gpu_ms += gpu_ms;

    let shared_activation_elements = carrier
        .n_tokens
        .checked_mul(carrier.ffn_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_llama_swiglu_f32_encode(
            ctx,
            enc,
            &carrier.gate_dev,
            &carrier.up_dev,
            &carrier.gate_dev,
            shared_activation_elements,
        )
    })?;
    trace.activation_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Activation, gpu_ms);
    total_gpu_ms += gpu_ms;

    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_shared_mul_mm_quant_f32_encode(
            ctx,
            enc,
            shared_quant.down,
            shared_down,
            shared_down_offset,
            &carrier.gate_dev,
            &carrier.shared_out_dev,
            carrier.ffn_dim,
            carrier.hidden_dim,
            carrier.n_tokens,
        )
    })?;
    trace.down_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Down, gpu_ms);
    total_gpu_ms += gpu_ms;

    let gpu_ms = qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        qwen_moe_llama_weighted_rank_reduce_f32_encode(
            ctx,
            enc,
            &carrier.shared_out_dev,
            &carrier.shared_route_weights_dev,
            &carrier.shared_out_dev,
            carrier.n_tokens,
            1,
            carrier.hidden_dim,
        )
    })?;
    trace.reduce_dispatches += 1;
    qwen_moe_llama_id_trace_add_stage(&mut trace, QwenMoeLlamaIdStage::Reduce, gpu_ms);
    total_gpu_ms += gpu_ms;

    let output_elements = carrier
        .n_tokens
        .checked_mul(carrier.hidden_dim)
        .ok_or(QwenMoeLlamaIdError::InvalidShape)?;
    total_gpu_ms += qwen_moe_llama_id_trace_split_stage(ctx, |enc| {
        encode_residual_add_inline_dim(
            ctx,
            enc,
            &carrier.sparse_moe_out_dev,
            &carrier.shared_out_dev,
            output_elements,
        )
    })?;
    Ok((trace, total_gpu_ms))
}

impl QwenMoeMulMmIdV3Scratch {
    pub(crate) fn checked(
        ffn_dim: usize,
        hidden_dim: usize,
        seq_len: usize,
        n_expert_used: usize,
        n_expert: usize,
        block_count: usize,
        budget_bytes: usize,
    ) -> Result<Self, QwenMoeIdPreflightReason> {
        if ffn_dim == 0 || hidden_dim == 0 || seq_len == 0 || n_expert_used == 0 || n_expert == 0 {
            return Err(QwenMoeIdPreflightReason::InvalidPrimitiveShape);
        }
        let slots = seq_len
            .checked_mul(n_expert_used)
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let gate_bytes = slots
            .checked_mul(ffn_dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let up_bytes = gate_bytes;
        let down_bytes = slots
            .checked_mul(hidden_dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let expert_meta = n_expert
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let block_meta = block_count
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let slot_meta = slots
            .checked_mul(3 * std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let metadata_bytes = expert_meta
            .checked_add(block_meta)
            .and_then(|v| v.checked_add(slot_meta))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let total_bytes = gate_bytes
            .checked_add(up_bytes)
            .and_then(|v| v.checked_add(down_bytes))
            .and_then(|v| v.checked_add(metadata_bytes))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        if total_bytes > budget_bytes {
            return Err(QwenMoeIdPreflightReason::ScratchOverBudget);
        }
        Ok(Self {
            gate_bytes,
            up_bytes,
            down_bytes,
            metadata_bytes,
            total_bytes,
        })
    }
}

pub(crate) fn qwen_moe_mulmmid_v3_dims_supported(hidden_dim: usize, ffn_dim: usize) -> bool {
    hidden_dim != 0 && ffn_dim != 0 && hidden_dim % 256 == 0 && ffn_dim % 256 == 0
}

impl QwenMoeMulMmIdV4Scratch {
    pub(crate) fn checked(
        ffn_dim: usize,
        hidden_dim: usize,
        seq_len: usize,
        n_expert_used: usize,
        n_expert: usize,
        block_count: usize,
        budget_bytes: usize,
    ) -> Result<Self, QwenMoeIdPreflightReason> {
        if ffn_dim == 0 || hidden_dim == 0 || seq_len == 0 || n_expert_used == 0 || n_expert == 0 {
            return Err(QwenMoeIdPreflightReason::InvalidPrimitiveShape);
        }
        let slots = seq_len
            .checked_mul(n_expert_used)
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let gate_bytes = slots
            .checked_mul(ffn_dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let up_bytes = gate_bytes;
        let down_bytes = 0;
        let expert_meta = n_expert
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let expert_rank_meta = n_expert
            .checked_mul(n_expert_used)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let block_meta = block_count
            .checked_mul(4)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let rank_meta = n_expert_used
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let slot_meta = slots
            .checked_mul(std::mem::size_of::<u32>() * 2 + std::mem::size_of::<f32>())
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let metadata_bytes = expert_meta
            .checked_add(expert_rank_meta)
            .and_then(|v| v.checked_add(block_meta))
            .and_then(|v| v.checked_add(rank_meta))
            .and_then(|v| v.checked_add(slot_meta))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        let total_bytes = gate_bytes
            .checked_add(up_bytes)
            .and_then(|v| v.checked_add(metadata_bytes))
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        if total_bytes > budget_bytes {
            return Err(QwenMoeIdPreflightReason::ScratchOverBudget);
        }
        Ok(Self {
            gate_bytes,
            up_bytes,
            down_bytes,
            metadata_bytes,
            total_bytes,
        })
    }
}

pub(crate) fn build_qwen_moe_raw_id_map_from_sorted(
    expert_ids: &[u32],
    token_ids: &[u32],
    n_expert: usize,
    seq_len: usize,
) -> Result<QwenMoeRawIdMap, QwenMoeIdPreflightReason> {
    if n_expert == 0 || seq_len == 0 || expert_ids.len() != token_ids.len() {
        return Err(QwenMoeIdPreflightReason::InvalidPrimitiveShape);
    }
    if expert_ids.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(QwenMoeIdPreflightReason::UnsortedExpertIds);
    }
    let map_len = n_expert
        .checked_mul(seq_len)
        .ok_or(QwenMoeIdPreflightReason::BlockCountOverflow)?;
    let mut expert_counts = vec![0u32; n_expert];
    let mut route_slot_ids = vec![u32::MAX; map_len];

    let mut group_start = 0usize;
    while group_start < expert_ids.len() {
        let expert = expert_ids[group_start] as usize;
        if expert >= n_expert {
            return Err(QwenMoeIdPreflightReason::ExpertIdOutOfRange);
        }
        let mut seen_tokens = vec![false; seq_len];
        let mut group_end = group_start;
        while group_end < expert_ids.len() && expert_ids[group_end] as usize == expert {
            let token = token_ids[group_end] as usize;
            if token >= seq_len {
                return Err(QwenMoeIdPreflightReason::TokenIdOutOfRange);
            }
            if seen_tokens[token] {
                return Err(QwenMoeIdPreflightReason::DuplicateTokenInExpertBlock);
            }
            seen_tokens[token] = true;
            let local = group_end - group_start;
            let dst = expert
                .checked_mul(seq_len)
                .and_then(|base| base.checked_add(local))
                .ok_or(QwenMoeIdPreflightReason::BlockCountOverflow)?;
            route_slot_ids[dst] = u32::try_from(group_end)
                .map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
            group_end += 1;
        }
        expert_counts[expert] = u32::try_from(group_end - group_start)
            .map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        group_start = group_end;
    }

    Ok(QwenMoeRawIdMap {
        expert_counts,
        route_slot_ids,
    })
}

pub(crate) fn qwen_moe_mulmmid_v4_dims_supported(hidden_dim: usize, ffn_dim: usize) -> bool {
    qwen_moe_mulmmid_v3_dims_supported(hidden_dim, ffn_dim)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_qwen_moe_mul_mm_id_v3_plan_token_major(
    expert_ids: &[u32],
    token_ids: &[u32],
    route_weights: &[f32],
    n_expert: usize,
    seq_len: usize,
    n_expert_used: usize,
    block_rows: usize,
) -> Result<QwenMoeMulMmIdV3Plan, QwenMoeIdPreflightReason> {
    if block_rows == 0 || n_expert == 0 || seq_len == 0 || !(1..=32).contains(&n_expert_used) {
        return Err(QwenMoeIdPreflightReason::InvalidPrimitiveShape);
    }
    let slots = seq_len
        .checked_mul(n_expert_used)
        .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
    if expert_ids.len() != slots || token_ids.len() != slots || route_weights.len() != slots {
        return Err(QwenMoeIdPreflightReason::RankCountMismatch);
    }

    for slot in 0..slots {
        let expected_token = slot / n_expert_used;
        let token = token_ids[slot] as usize;
        if token >= seq_len {
            return Err(QwenMoeIdPreflightReason::TokenIdOutOfRange);
        }
        if token != expected_token {
            return Err(QwenMoeIdPreflightReason::RankCountMismatch);
        }
        if expert_ids[slot] as usize >= n_expert {
            return Err(QwenMoeIdPreflightReason::ExpertIdOutOfRange);
        }
    }

    let mut order: Vec<usize> = (0..slots).collect();
    order.sort_unstable_by_key(|&idx| (expert_ids[idx], token_ids[idx], idx % n_expert_used));

    let mut expert_offsets = vec![0u32; n_expert];
    let mut expert_counts = vec![0u32; n_expert];
    let mut block_experts = Vec::new();
    let mut block_local0 = Vec::new();
    let mut token_ids_sorted = Vec::with_capacity(slots);
    let mut rank_ids_sorted = Vec::with_capacity(slots);
    let mut dst_slots_sorted = Vec::with_capacity(slots);
    let mut route_weights_sorted = Vec::with_capacity(slots);
    let mut last_expert_token: Option<(u32, u32)> = None;
    let mut sorted_pos = 0usize;

    while sorted_pos < slots {
        let expert = expert_ids[order[sorted_pos]] as usize;
        let group_start = sorted_pos;
        while sorted_pos < slots && expert_ids[order[sorted_pos]] as usize == expert {
            let idx = order[sorted_pos];
            let token = token_ids[idx];
            let rank = (idx % n_expert_used) as u32;
            if last_expert_token == Some((expert as u32, token)) {
                return Err(QwenMoeIdPreflightReason::DuplicateTokenInExpertBlock);
            }
            last_expert_token = Some((expert as u32, token));
            token_ids_sorted.push(token);
            rank_ids_sorted.push(rank);
            dst_slots_sorted.push(token * n_expert_used as u32 + rank);
            route_weights_sorted.push(route_weights[idx]);
            sorted_pos += 1;
        }
        let count = sorted_pos - group_start;
        expert_offsets[expert] =
            u32::try_from(group_start).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        expert_counts[expert] =
            u32::try_from(count).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        for local0 in (0..count).step_by(block_rows) {
            block_experts.push(
                u32::try_from(expert).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?,
            );
            block_local0.push(
                u32::try_from(local0).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?,
            );
        }
    }

    Ok(QwenMoeMulMmIdV3Plan {
        expert_offsets,
        expert_counts,
        block_experts,
        block_local0,
        token_ids_sorted,
        rank_ids_sorted,
        dst_slots_sorted,
        route_weights_sorted,
        route_weights_token_rank: route_weights.to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_qwen_moe_mul_mm_id_v4_plan_token_major(
    expert_ids: &[u32],
    token_ids: &[u32],
    route_weights: &[f32],
    n_expert: usize,
    seq_len: usize,
    n_expert_used: usize,
    block_rows: usize,
) -> Result<QwenMoeMulMmIdV4Plan, QwenMoeIdPreflightReason> {
    if block_rows == 0 || n_expert == 0 || seq_len == 0 || !(1..=32).contains(&n_expert_used) {
        return Err(QwenMoeIdPreflightReason::InvalidPrimitiveShape);
    }
    let slots = seq_len
        .checked_mul(n_expert_used)
        .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
    if expert_ids.len() != slots || token_ids.len() != slots || route_weights.len() != slots {
        return Err(QwenMoeIdPreflightReason::RankCountMismatch);
    }

    let mut seen_expert_token = vec![false; n_expert * seq_len];
    for slot in 0..slots {
        let expected_token = slot / n_expert_used;
        let token = token_ids[slot] as usize;
        if token >= seq_len {
            return Err(QwenMoeIdPreflightReason::TokenIdOutOfRange);
        }
        if token != expected_token {
            return Err(QwenMoeIdPreflightReason::RankCountMismatch);
        }
        let expert = expert_ids[slot] as usize;
        if expert >= n_expert {
            return Err(QwenMoeIdPreflightReason::ExpertIdOutOfRange);
        }
        let seen = &mut seen_expert_token[expert * seq_len + token];
        if *seen {
            return Err(QwenMoeIdPreflightReason::DuplicateTokenInExpertBlock);
        }
        *seen = true;
    }

    let mut order: Vec<usize> = (0..slots).collect();
    order.sort_unstable_by_key(|&idx| (expert_ids[idx], idx % n_expert_used, token_ids[idx]));

    let mut expert_offsets = vec![0u32; n_expert];
    let mut expert_counts = vec![0u32; n_expert];
    let mut id_block_experts = Vec::new();
    let mut id_block_local0 = Vec::new();
    let mut expert_rank_offsets = vec![0u32; n_expert * n_expert_used];
    let mut expert_rank_counts = vec![0u32; n_expert * n_expert_used];
    let mut rank_block_offsets = vec![0u32; n_expert_used];
    let mut rank_block_counts = vec![0u32; n_expert_used];
    let mut rank_block_experts = Vec::new();
    let mut rank_block_local0 = Vec::new();
    let mut token_ids_sorted = Vec::with_capacity(slots);
    let mut rank_ids_sorted = Vec::with_capacity(slots);
    let mut route_weights_sorted = Vec::with_capacity(slots);

    for &idx in &order {
        token_ids_sorted.push(token_ids[idx]);
        rank_ids_sorted.push((idx % n_expert_used) as u32);
        route_weights_sorted.push(route_weights[idx]);
    }

    let mut pos = 0usize;
    for expert in 0..n_expert {
        let expert_start = pos;
        for rank in 0..n_expert_used {
            let rank_start = pos;
            while pos < slots
                && expert_ids[order[pos]] as usize == expert
                && order[pos] % n_expert_used == rank
            {
                pos += 1;
            }
            let count = pos - rank_start;
            let pair = expert * n_expert_used + rank;
            expert_rank_offsets[pair] = u32::try_from(rank_start)
                .map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
            expert_rank_counts[pair] =
                u32::try_from(count).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        }
        let expert_count = pos - expert_start;
        expert_offsets[expert] = u32::try_from(expert_start)
            .map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        expert_counts[expert] = u32::try_from(expert_count)
            .map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        for local0 in (0..expert_count).step_by(block_rows) {
            id_block_experts.push(
                u32::try_from(expert).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?,
            );
            id_block_local0.push(
                u32::try_from(local0).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?,
            );
        }
    }

    for rank in 0..n_expert_used {
        let start = rank_block_experts.len();
        for expert in 0..n_expert {
            let pair = expert * n_expert_used + rank;
            let count = expert_rank_counts[pair] as usize;
            if count == 0 {
                continue;
            }
            for local0 in (0..count).step_by(block_rows) {
                rank_block_experts.push(
                    u32::try_from(expert)
                        .map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?,
                );
                rank_block_local0.push(
                    u32::try_from(local0)
                        .map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?,
                );
            }
        }
        rank_block_offsets[rank] =
            u32::try_from(start).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        rank_block_counts[rank] = u32::try_from(rank_block_experts.len() - start)
            .map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
    }

    Ok(QwenMoeMulMmIdV4Plan {
        expert_offsets,
        expert_counts,
        id_block_experts,
        id_block_local0,
        expert_rank_offsets,
        expert_rank_counts,
        rank_block_offsets,
        rank_block_counts,
        rank_block_experts,
        rank_block_local0,
        token_ids_sorted,
        rank_ids_sorted,
        route_weights_sorted,
    })
}

pub(crate) const QWEN_MOE_ID_PRIMITIVE_TILE_SLOTS: usize = 64;
pub(crate) const QWEN_MOE_ID_PRIMITIVE_FFN_TILE: usize = 256;
pub(crate) const QWEN_MOE_ID_PRIMITIVE_HIDDEN_TILE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenMoeIdPrimitiveShape {
    pub(crate) tile_slots: usize,
    pub(crate) ffn_tile: usize,
    pub(crate) hidden_tile: usize,
    pub(crate) ffn_tiles_per_pass: usize,
}

impl QwenMoeIdPrimitiveShape {
    pub(crate) fn scratch_bytes(self) -> usize {
        let f32_bytes = std::mem::size_of::<f32>();
        let gate_up_act = self.tile_slots * self.ffn_tile * 3;
        (gate_up_act * f32_bytes) + self.partial_bytes()
    }

    pub(crate) fn partial_bytes(self) -> usize {
        self.ffn_tiles_per_pass * self.tile_slots * self.hidden_tile * std::mem::size_of::<f32>()
    }
}

pub(crate) fn qwen_moe_id_primitive_predicted_dispatch_count(
    block_count: usize,
    hidden_dim: usize,
    shape: QwenMoeIdPrimitiveShape,
) -> Result<usize, QwenMoeIdPreflightReason> {
    if shape.hidden_tile == 0 || shape.ffn_tiles_per_pass == 0 {
        return Err(QwenMoeIdPreflightReason::InvalidPrimitiveShape);
    }
    let hidden_tiles = hidden_dim.div_ceil(shape.hidden_tile);
    let per_hidden_tile = shape
        .ffn_tiles_per_pass
        .checked_mul(3)
        .and_then(|v| v.checked_add(1))
        .ok_or(QwenMoeIdPreflightReason::BlockCountOverflow)?;
    block_count
        .checked_mul(hidden_tiles)
        .and_then(|v| v.checked_mul(per_hidden_tile))
        .ok_or(QwenMoeIdPreflightReason::BlockCountOverflow)
}

pub(crate) fn qwen_moe_id_primitive_dispatch_budget(
    block_count: usize,
    hidden_dim: usize,
    shape: QwenMoeIdPrimitiveShape,
) -> Result<usize, QwenMoeIdPreflightReason> {
    if shape.hidden_tile == 0 || shape.ffn_tiles_per_pass == 0 {
        return Err(QwenMoeIdPreflightReason::InvalidPrimitiveShape);
    }
    let hidden_tiles = hidden_dim.div_ceil(shape.hidden_tile);
    let per_block = shape
        .ffn_tiles_per_pass
        .checked_mul(3)
        .and_then(|v| v.checked_add(hidden_tiles))
        .and_then(|v| v.checked_add(1))
        .and_then(|v| v.checked_mul(4))
        .ok_or(QwenMoeIdPreflightReason::BlockCountOverflow)?;
    block_count
        .checked_mul(per_block)
        .ok_or(QwenMoeIdPreflightReason::BlockCountOverflow)
}

pub(crate) fn build_qwen_moe_id_execution_plan(
    expert_ids: &[u32],
    token_ids: &[u32],
    route_weights: &[f32],
    n_expert: usize,
    seq_len: usize,
    block_rows: usize,
) -> Result<QwenMoeIdExecutionPlan, QwenMoeIdPreflightReason> {
    if block_rows == 0 {
        return Err(QwenMoeIdPreflightReason::InvalidExpertArena);
    }
    if expert_ids.len() != token_ids.len() || expert_ids.len() != route_weights.len() {
        return Err(QwenMoeIdPreflightReason::InvalidExpertArena);
    }
    if !expert_ids.windows(2).all(|pair| pair[0] <= pair[1]) {
        return Err(QwenMoeIdPreflightReason::UnsortedExpertIds);
    }

    let mut expert_offsets = vec![0u32; n_expert];
    let mut expert_counts = vec![0u32; n_expert];
    let mut block_experts = Vec::new();
    let mut block_local0 = Vec::new();
    let mut group_start = 0usize;

    while group_start < expert_ids.len() {
        let expert = expert_ids[group_start] as usize;
        if expert >= n_expert {
            return Err(QwenMoeIdPreflightReason::ExpertIdOutOfRange);
        }
        let mut group_end = group_start + 1;
        while group_end < expert_ids.len() && expert_ids[group_end] as usize == expert {
            group_end += 1;
        }
        for &token in &token_ids[group_start..group_end] {
            if token as usize >= seq_len {
                return Err(QwenMoeIdPreflightReason::TokenIdOutOfRange);
            }
        }
        let count = group_end - group_start;
        let mut seen_tokens = std::collections::HashSet::with_capacity(count);
        for &token in &token_ids[group_start..group_end] {
            if !seen_tokens.insert(token) {
                return Err(QwenMoeIdPreflightReason::DuplicateTokenInExpertBlock);
            }
        }
        expert_offsets[expert] =
            u32::try_from(group_start).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        expert_counts[expert] =
            u32::try_from(count).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?;
        for local0 in (0..count).step_by(block_rows) {
            block_experts.push(
                u32::try_from(expert).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?,
            );
            block_local0.push(
                u32::try_from(local0).map_err(|_| QwenMoeIdPreflightReason::BlockCountOverflow)?,
            );
        }
        group_start = group_end;
    }

    Ok(QwenMoeIdExecutionPlan {
        expert_offsets,
        expert_counts,
        block_experts,
        block_local0,
        token_ids: token_ids.to_vec(),
        route_weights: route_weights.to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn preflight_qwen_moe_id_primitive_inputs(
    pipelines_available: bool,
    down_is_q6k: bool,
    gate_all_len: usize,
    up_all_len: usize,
    down_all_len: usize,
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    expert_ids: &[u32],
    token_ids: &[u32],
    route_weights: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    block_rows: usize,
    shape: QwenMoeIdPrimitiveShape,
    partial_budget_bytes: usize,
) -> Result<QwenMoeIdExecutionPlan, QwenMoeIdPreflightReason> {
    if !pipelines_available {
        return Err(QwenMoeIdPreflightReason::PipelineUnavailable);
    }
    if !down_is_q6k {
        return Err(QwenMoeIdPreflightReason::UnsupportedDownQuant);
    }
    if gate_expert_bytes == 0
        || up_expert_bytes == 0
        || down_expert_bytes == 0
        || gate_all_len % gate_expert_bytes != 0
        || up_all_len % up_expert_bytes != 0
        || down_all_len % down_expert_bytes != 0
    {
        return Err(QwenMoeIdPreflightReason::InvalidExpertArena);
    }
    let n_gate = gate_all_len / gate_expert_bytes;
    let n_up = up_all_len / up_expert_bytes;
    let n_down = down_all_len / down_expert_bytes;
    if n_gate != n_up || n_gate != n_down {
        return Err(QwenMoeIdPreflightReason::ExpertArenaMismatch);
    }
    if shape.tile_slots == 0
        || shape.ffn_tile == 0
        || shape.hidden_tile == 0
        || shape.ffn_tile > ffn_dim
        || shape.hidden_tile > hidden_dim
        || shape.ffn_tiles_per_pass != ffn_dim.div_ceil(shape.ffn_tile)
    {
        return Err(QwenMoeIdPreflightReason::InvalidPrimitiveShape);
    }
    if shape.partial_bytes() > partial_budget_bytes {
        return Err(QwenMoeIdPreflightReason::PartialBufferOverBudget);
    }

    build_qwen_moe_id_execution_plan(
        expert_ids,
        token_ids,
        route_weights,
        n_gate,
        seq_len,
        block_rows,
    )
}

impl QwenMoePrefillAccumCarrier {
    pub(crate) fn new(ctx: &MetalContext, hidden_dim: usize, seq_len: usize) -> Self {
        Self {
            hidden_dim,
            seq_len,
            out_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            norm_all_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            hidden_buf: u32_buf(ctx, hidden_dim as u32),
        }
    }

    pub(crate) fn clear(&self) {
        let bytes = self.seq_len * self.hidden_dim * std::mem::size_of::<f32>();
        let contents = self.out_dev.contents();
        unsafe {
            std::ptr::write_bytes(contents.as_ptr() as *mut u8, 0, bytes);
        }
    }

    pub(crate) fn readback(&self) -> Vec<f32> {
        readback(&self.out_dev, self.seq_len * self.hidden_dim)
    }

    pub(crate) fn upload_norm_all(&self, norm_all: &[f32]) {
        debug_assert_eq!(norm_all.len(), self.seq_len * self.hidden_dim);
        let contents = self.norm_all_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                norm_all.as_ptr(),
                contents.as_ptr() as *mut f32,
                self.seq_len * self.hidden_dim,
            );
        }
    }
}

impl QwenMoePrefillIdGateUpCarrier {
    pub(crate) fn new(ctx: &MetalContext, slots: usize, ffn_dim: usize) -> Self {
        Self {
            slots,
            ffn_dim,
            gate_dev: private_f32_buf(ctx, slots * ffn_dim),
            up_dev: private_f32_buf(ctx, slots * ffn_dim),
        }
    }

    pub(crate) fn scratch_bytes(&self) -> usize {
        let f32_bytes = std::mem::size_of::<f32>();
        self.slots * self.ffn_dim * 2 * f32_bytes
    }
}

impl QwenMoePrefillIdGateUpF16Carrier {
    pub(crate) fn new(ctx: &MetalContext, slots: usize, ffn_dim: usize) -> Self {
        Self {
            slots,
            ffn_dim,
            gate_dev: private_f16_buf(ctx, slots * ffn_dim),
            up_dev: private_f16_buf(ctx, slots * ffn_dim),
        }
    }

    pub(crate) fn scratch_bytes(&self) -> usize {
        let f16_bytes = std::mem::size_of::<u16>();
        self.slots * self.ffn_dim * 2 * f16_bytes
    }
}

impl QwenMoeMulMmIdV3Carrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        n_expert_used: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        n_expert: usize,
        block_count: usize,
        budget_bytes: usize,
    ) -> Result<Self, QwenMoeIdPreflightReason> {
        let scratch = QwenMoeMulMmIdV3Scratch::checked(
            ffn_dim,
            hidden_dim,
            seq_len,
            n_expert_used,
            n_expert,
            block_count,
            budget_bytes,
        )?;
        let slots = seq_len
            .checked_mul(n_expert_used)
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        Ok(Self {
            slots,
            seq_len,
            n_expert_used,
            hidden_dim,
            ffn_dim,
            gate_half_dev: private_f16_buf(ctx, slots * ffn_dim),
            up_half_dev: private_f16_buf(ctx, slots * ffn_dim),
            down_token_rank_dev: private_f32_buf(ctx, slots * hidden_dim),
            scratch,
        })
    }

    pub(crate) fn scratch(&self) -> QwenMoeMulMmIdV3Scratch {
        self.scratch
    }
}

impl QwenMoeMulMmIdV4Carrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        n_expert_used: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        n_expert: usize,
        block_count: usize,
        budget_bytes: usize,
    ) -> Result<Self, QwenMoeIdPreflightReason> {
        let scratch = QwenMoeMulMmIdV4Scratch::checked(
            ffn_dim,
            hidden_dim,
            seq_len,
            n_expert_used,
            n_expert,
            block_count,
            budget_bytes,
        )?;
        let slots = seq_len
            .checked_mul(n_expert_used)
            .ok_or(QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
        Ok(Self {
            slots,
            n_expert_used,
            hidden_dim,
            ffn_dim,
            gate_half_dev: private_f16_buf(ctx, slots * ffn_dim),
            up_half_dev: private_f16_buf(ctx, slots * ffn_dim),
            scratch,
        })
    }

    pub(crate) fn scratch(&self) -> QwenMoeMulMmIdV4Scratch {
        self.scratch
    }

    #[cfg(test)]
    pub(crate) fn gate_bytes(&self) -> usize {
        self.scratch.gate_bytes
    }

    #[cfg(test)]
    pub(crate) fn up_bytes(&self) -> usize {
        self.scratch.up_bytes
    }

    #[cfg(test)]
    pub(crate) fn metadata_bytes(&self) -> usize {
        self.scratch.metadata_bytes
    }

    #[cfg(test)]
    pub(crate) fn down_scratch_bytes(&self) -> usize {
        self.scratch.down_bytes
    }

    #[cfg(test)]
    pub(crate) fn scratch_bytes(&self) -> usize {
        self.scratch.total_bytes
    }
}

impl QwenMoePrefillIdCarrier {
    pub(crate) fn new(ctx: &MetalContext, slots: usize, hidden_dim: usize, ffn_dim: usize) -> Self {
        Self {
            slots,
            hidden_dim,
            ffn_dim,
            gate_dev: private_f32_buf(ctx, slots * ffn_dim),
            up_dev: private_f32_buf(ctx, slots * ffn_dim),
            down_dev: private_f32_buf(ctx, slots * hidden_dim),
            total_ff_buf: u32_buf(ctx, (slots * ffn_dim) as u32),
        }
    }

    pub(crate) fn scratch_bytes(&self) -> usize {
        let f32_bytes = std::mem::size_of::<f32>();
        (self.slots * self.ffn_dim * 2 + self.slots * self.hidden_dim) * f32_bytes
    }
}

pub(crate) struct QwenMoeDecodeCarrier {
    slots: usize,
    n_embd: usize,
    n_ff: usize,
    input_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    route_weights_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_embd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_ff_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    slots_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_ff_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_off_buf: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    up_off_buf: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    down_off_buf: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl QwenMoeDecodeCarrier {
    pub(crate) fn new(ctx: &MetalContext, slots: usize, n_embd: usize, n_ff: usize) -> Self {
        let shared = MTLResourceOptions::StorageModeShared;
        let gate_off_buf = (0..slots).map(|_| u32_buf(ctx, 0)).collect();
        let up_off_buf = (0..slots).map(|_| u32_buf(ctx, 0)).collect();
        let down_off_buf = (0..slots).map(|_| u32_buf(ctx, 0)).collect();
        Self {
            slots,
            n_embd,
            n_ff,
            input_dev: empty_f32_buf(ctx, n_embd),
            gate_dev: empty_f32_buf(ctx, slots * n_ff),
            up_dev: empty_f32_buf(ctx, slots * n_ff),
            down_dev: empty_f32_buf(ctx, slots * n_embd),
            out_dev: empty_f32_buf(ctx, n_embd),
            route_weights_dev: ctx
                .device
                .newBufferWithLength_options(slots * std::mem::size_of::<f32>(), shared)
                .expect("Qwen MoE decode route weight buffer"),
            n_embd_buf: u32_buf(ctx, n_embd as u32),
            n_ff_buf: u32_buf(ctx, n_ff as u32),
            slots_buf: u32_buf(ctx, slots as u32),
            total_ff_buf: u32_buf(ctx, (slots * n_ff) as u32),
            gate_off_buf,
            up_off_buf,
            down_off_buf,
        }
    }

    fn upload_input(&self, input: &[f32]) {
        debug_assert_eq!(input.len(), self.n_embd);
        let contents = self.input_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                contents.as_ptr() as *mut f32,
                self.n_embd,
            );
        }
    }

    fn upload_offsets(&self, gate_off: &[u32], up_off: &[u32], down_off: &[u32]) {
        debug_assert_eq!(gate_off.len(), self.slots);
        debug_assert_eq!(up_off.len(), self.slots);
        debug_assert_eq!(down_off.len(), self.slots);
        fn store(buf: &ProtocolObject<dyn MTLBuffer>, value: u32) {
            unsafe {
                std::ptr::write(buf.contents().as_ptr() as *mut u32, value);
            }
        }
        for slot in 0..self.slots {
            store(&self.gate_off_buf[slot], gate_off[slot]);
            store(&self.up_off_buf[slot], up_off[slot]);
            store(&self.down_off_buf[slot], down_off[slot]);
        }
    }

    fn upload_route_weights(&self, route_weights: &[f32]) {
        debug_assert_eq!(route_weights.len(), self.slots);
        unsafe {
            std::ptr::copy_nonoverlapping(
                route_weights.as_ptr(),
                self.route_weights_dev.contents().as_ptr() as *mut f32,
                self.slots,
            );
        }
    }
}

pub(crate) struct QwenMoeDecodeIdCarrier {
    slots: usize,
    n_embd: usize,
    n_ff: usize,
    input_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    expert_ids_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    route_weights_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_embd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_ff_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    slots_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_ff_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    per_gate_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    per_up_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    per_down_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    shared_expert_id_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_sparse_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_sparse_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_sparse_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_shared_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_shared_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_shared_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct QwenMoeDecodeTableKey {
    gate_sparse: (usize, usize),
    up_sparse: (usize, usize),
    down_sparse: (usize, usize),
    gate_shared: (usize, usize),
    up_shared: (usize, usize),
    down_shared: (usize, usize),
}

struct QwenMoeArgumentWeightTable {
    argument_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    weights: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    weight_resources: Vec<NonNull<ProtocolObject<dyn MTLResource>>>,
}

struct QwenMoeDecodeTableSet {
    gate: QwenMoeArgumentWeightTable,
    up: QwenMoeArgumentWeightTable,
    down: QwenMoeArgumentWeightTable,
}

impl QwenMoeDecodeIdCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        slots: usize,
        n_embd: usize,
        n_ff: usize,
        per_gate: usize,
        per_up: usize,
        per_down: usize,
        shared_expert_id: u32,
    ) -> Self {
        let shared = MTLResourceOptions::StorageModeShared;
        Self {
            slots,
            n_embd,
            n_ff,
            input_dev: empty_f32_buf(ctx, n_embd),
            gate_dev: empty_f32_buf(ctx, slots * n_ff),
            up_dev: empty_f32_buf(ctx, slots * n_ff),
            down_dev: empty_f32_buf(ctx, slots * n_embd),
            out_dev: empty_f32_buf(ctx, n_embd),
            expert_ids_dev: ctx
                .device
                .newBufferWithLength_options(slots * std::mem::size_of::<u32>(), shared)
                .expect("Qwen MoE decode expert id buffer"),
            route_weights_dev: ctx
                .device
                .newBufferWithLength_options(slots * std::mem::size_of::<f32>(), shared)
                .expect("Qwen MoE decode route weight buffer"),
            n_embd_buf: u32_buf(ctx, n_embd as u32),
            n_ff_buf: u32_buf(ctx, n_ff as u32),
            slots_buf: u32_buf(ctx, slots as u32),
            total_ff_buf: u32_buf(ctx, (slots * n_ff) as u32),
            per_gate_buf: u64_buf(ctx, per_gate as u64),
            per_up_buf: u64_buf(ctx, per_up as u64),
            per_down_buf: u64_buf(ctx, per_down as u64),
            shared_expert_id_buf: u32_buf(ctx, shared_expert_id),
            gate_sparse_off_buf: u64_buf(ctx, 0),
            up_sparse_off_buf: u64_buf(ctx, 0),
            down_sparse_off_buf: u64_buf(ctx, 0),
            gate_shared_off_buf: u64_buf(ctx, 0),
            up_shared_off_buf: u64_buf(ctx, 0),
            down_shared_off_buf: u64_buf(ctx, 0),
        }
    }

    fn upload_input(&self, input: &[f32]) {
        debug_assert_eq!(input.len(), self.n_embd);
        let contents = self.input_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                contents.as_ptr() as *mut f32,
                self.n_embd,
            );
        }
    }

    fn upload_routes(&self, expert_ids: &[u32], route_weights: &[f32]) {
        debug_assert_eq!(expert_ids.len(), self.slots);
        debug_assert_eq!(route_weights.len(), self.slots);
        unsafe {
            std::ptr::copy_nonoverlapping(
                expert_ids.as_ptr(),
                self.expert_ids_dev.contents().as_ptr() as *mut u32,
                self.slots,
            );
            std::ptr::copy_nonoverlapping(
                route_weights.as_ptr(),
                self.route_weights_dev.contents().as_ptr() as *mut f32,
                self.slots,
            );
        }
    }

    fn upload_offsets(
        &self,
        gate_sparse_off: u32,
        up_sparse_off: u32,
        down_sparse_off: u32,
        gate_shared_off: u32,
        up_shared_off: u32,
        down_shared_off: u32,
    ) {
        fn store(buf: &ProtocolObject<dyn MTLBuffer>, value: u32) {
            unsafe {
                std::ptr::write(buf.contents().as_ptr() as *mut u64, value as u64);
            }
        }
        store(&self.gate_sparse_off_buf, gate_sparse_off);
        store(&self.up_sparse_off_buf, up_sparse_off);
        store(&self.down_sparse_off_buf, down_sparse_off);
        store(&self.gate_shared_off_buf, gate_shared_off);
        store(&self.up_shared_off_buf, up_shared_off);
        store(&self.down_shared_off_buf, down_shared_off);
    }
}

pub(crate) struct QwenMoeDecodeChainCarrier {
    slots: usize,
    n_embd: usize,
    n_ff: usize,
    n_expert: usize,
    n_used: usize,
    per_gate: usize,
    per_up: usize,
    per_down: usize,
    normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    router_logits_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    moe_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    expert_ids_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    route_weights_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_embd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_ff_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_expert_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    n_used_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    slots_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_ff_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    per_gate_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    per_up_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    per_down_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    shared_expert_id_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_sparse_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_sparse_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_sparse_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_shared_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    up_shared_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    down_shared_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    selected_slice_cache:
        RefCell<HashMap<(usize, usize), (Retained<ProtocolObject<dyn MTLBuffer>>, u32)>>,
    table_cache: RefCell<HashMap<QwenMoeDecodeTableKey, QwenMoeDecodeTableSet>>,
}

impl QwenMoeDecodeChainCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        n_used: usize,
        n_embd: usize,
        n_ff: usize,
        n_expert: usize,
        per_gate: usize,
        per_up: usize,
        per_down: usize,
        shared_expert_id: u32,
        eps: f32,
    ) -> Self {
        let slots = n_used + 1;
        let shared = MTLResourceOptions::StorageModeShared;
        Self {
            slots,
            n_embd,
            n_ff,
            n_expert,
            n_used,
            per_gate,
            per_up,
            per_down,
            normed_dev: empty_f32_buf(ctx, n_embd),
            router_logits_dev: empty_f32_buf(ctx, n_expert),
            gate_dev: empty_f32_buf(ctx, slots * n_ff),
            up_dev: empty_f32_buf(ctx, slots * n_ff),
            down_dev: empty_f32_buf(ctx, slots * n_embd),
            moe_out_dev: empty_f32_buf(ctx, n_embd),
            expert_ids_dev: ctx
                .device
                .newBufferWithLength_options(slots * std::mem::size_of::<u32>(), shared)
                .expect("Qwen MoE decode chain expert id buffer"),
            route_weights_dev: ctx
                .device
                .newBufferWithLength_options(slots * std::mem::size_of::<f32>(), shared)
                .expect("Qwen MoE decode chain route weight buffer"),
            n_embd_buf: u32_buf(ctx, n_embd as u32),
            n_ff_buf: u32_buf(ctx, n_ff as u32),
            n_expert_buf: u32_buf(ctx, n_expert as u32),
            n_used_buf: u32_buf(ctx, n_used as u32),
            eps_buf: f32_buf(ctx, eps),
            slots_buf: u32_buf(ctx, slots as u32),
            total_ff_buf: u32_buf(ctx, (slots * n_ff) as u32),
            per_gate_buf: u64_buf(ctx, per_gate as u64),
            per_up_buf: u64_buf(ctx, per_up as u64),
            per_down_buf: u64_buf(ctx, per_down as u64),
            shared_expert_id_buf: u32_buf(ctx, shared_expert_id),
            gate_sparse_off_buf: u64_buf(ctx, 0),
            up_sparse_off_buf: u64_buf(ctx, 0),
            down_sparse_off_buf: u64_buf(ctx, 0),
            gate_shared_off_buf: u64_buf(ctx, 0),
            up_shared_off_buf: u64_buf(ctx, 0),
            down_shared_off_buf: u64_buf(ctx, 0),
            selected_slice_cache: RefCell::new(HashMap::new()),
            table_cache: RefCell::new(HashMap::new()),
        }
    }

    fn upload_offsets(
        &self,
        gate_sparse_off: u32,
        up_sparse_off: u32,
        down_sparse_off: u32,
        gate_shared_off: u32,
        up_shared_off: u32,
        down_shared_off: u32,
    ) {
        fn store(buf: &ProtocolObject<dyn MTLBuffer>, value: u32) {
            unsafe {
                std::ptr::write(buf.contents().as_ptr() as *mut u64, value as u64);
            }
        }
        store(&self.gate_sparse_off_buf, gate_sparse_off);
        store(&self.up_sparse_off_buf, up_sparse_off);
        store(&self.down_sparse_off_buf, down_sparse_off);
        store(&self.gate_shared_off_buf, gate_shared_off);
        store(&self.up_shared_off_buf, up_shared_off);
        store(&self.down_shared_off_buf, down_shared_off);
    }

    pub(crate) fn upload_routes(&self, expert_ids: &[u32], route_weights: &[f32]) {
        debug_assert_eq!(expert_ids.len(), self.slots);
        debug_assert_eq!(route_weights.len(), self.slots);
        unsafe {
            std::ptr::copy_nonoverlapping(
                expert_ids.as_ptr(),
                self.expert_ids_dev.contents().as_ptr() as *mut u32,
                self.slots,
            );
            std::ptr::copy_nonoverlapping(
                route_weights.as_ptr(),
                self.route_weights_dev.contents().as_ptr() as *mut f32,
                self.slots,
            );
        }
    }

    pub(crate) fn n_used(&self) -> usize {
        self.n_used
    }

    pub(crate) fn n_expert(&self) -> usize {
        self.n_expert
    }

    pub(crate) fn per_gate(&self) -> usize {
        self.per_gate
    }

    pub(crate) fn per_up(&self) -> usize {
        self.per_up
    }

    pub(crate) fn per_down(&self) -> usize {
        self.per_down
    }

    pub(crate) fn read_router_logits(&self) -> Vec<f32> {
        readback(&self.router_logits_dev, self.n_expert)
    }

    pub(crate) fn wrap_selected_slice(
        &self,
        ctx: &MetalContext,
        raw: &[u8],
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, u32) {
        let key = (raw.as_ptr() as usize, raw.len());
        let mut cache = self.selected_slice_cache.borrow_mut();
        let entry = cache
            .entry(key)
            .or_insert_with(|| crate::compute::wrap_nocopy(ctx, raw));
        (entry.0.clone(), entry.1)
    }

    fn build_argument_weight_table(
        &self,
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
        sparse_raw: &[u8],
        shared_raw: &[u8],
        per_expert: usize,
    ) -> Option<QwenMoeArgumentWeightTable> {
        if self.n_expert + 1 > QWEN_MOE_WEIGHT_TABLE_CAP
            || per_expert == 0
            || sparse_raw.len() < self.n_expert.saturating_mul(per_expert)
        {
            return None;
        }

        let chunk_count = self.n_expert.div_ceil(QWEN_MOE_WEIGHT_CHUNK_EXPERTS);
        let mut chunks = Vec::with_capacity(chunk_count);
        for chunk in 0..chunk_count {
            let start = chunk * QWEN_MOE_WEIGHT_CHUNK_EXPERTS * per_expert;
            let expert_count = (self.n_expert - chunk * QWEN_MOE_WEIGHT_CHUNK_EXPERTS)
                .min(QWEN_MOE_WEIGHT_CHUNK_EXPERTS);
            let len = expert_count * per_expert;
            let (weight, offset) = self.wrap_selected_slice(ctx, &sparse_raw[start..start + len]);
            chunks.push((weight, offset as usize));
        }
        let (shared_weight, shared_offset) = self.wrap_selected_slice(ctx, shared_raw);

        let mut table_weights = Vec::with_capacity(QWEN_MOE_WEIGHT_TABLE_CAP);
        let mut offsets = Vec::with_capacity(QWEN_MOE_WEIGHT_TABLE_CAP);
        for expert in 0..QWEN_MOE_WEIGHT_TABLE_CAP {
            let (weight, offset) = if expert < self.n_expert {
                let chunk = expert / QWEN_MOE_WEIGHT_CHUNK_EXPERTS;
                let local = expert % QWEN_MOE_WEIGHT_CHUNK_EXPERTS;
                let (chunk_weight, chunk_offset) = &chunks[chunk];
                (chunk_weight.clone(), *chunk_offset + local * per_expert)
            } else {
                (shared_weight.clone(), shared_offset as usize)
            };
            table_weights.push(weight);
            offsets.push(offset);
        }

        let argument_buf = ctx
            .device
            .newBufferWithLength_options(
                encoder.encodedLength() as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("Qwen MoE argument weight table");
        unsafe {
            encoder.setArgumentBuffer_offset(Some(&argument_buf), 0);
            let mut buffer_ptrs: Vec<*const ProtocolObject<dyn MTLBuffer>> = table_weights
                .iter()
                .map(|buffer| &**buffer as *const _)
                .collect();
            encoder.setBuffers_offsets_withRange(
                NonNull::new(buffer_ptrs.as_mut_ptr()).expect("Qwen MoE table buffer ptrs"),
                NonNull::new(offsets.as_mut_ptr()).expect("Qwen MoE table offsets"),
                NSRange {
                    location: 0,
                    length: table_weights.len(),
                },
            );
        }
        let mut weights: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> =
            chunks.into_iter().map(|(weight, _)| weight).collect();
        weights.push(shared_weight);
        let weight_resources = weights
            .iter()
            .map(|buffer| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&**buffer)))
            .collect();
        Some(QwenMoeArgumentWeightTable {
            argument_buf,
            weights,
            weight_resources,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_table_set(
        &self,
        ctx: &MetalContext,
        down_quant: u8,
        gate_sparse_raw: &[u8],
        up_sparse_raw: &[u8],
        down_sparse_raw: &[u8],
        gate_shared_raw: &[u8],
        up_shared_raw: &[u8],
        down_shared_raw: &[u8],
    ) -> Option<QwenMoeDecodeTableSet> {
        Some(QwenMoeDecodeTableSet {
            gate: self.build_argument_weight_table(
                ctx,
                ctx.qwen_moe_decode_q4k_table_arg_encoder.as_ref()?,
                gate_sparse_raw,
                gate_shared_raw,
                self.per_gate,
            )?,
            up: self.build_argument_weight_table(
                ctx,
                ctx.qwen_moe_decode_q4k_table_arg_encoder.as_ref()?,
                up_sparse_raw,
                up_shared_raw,
                self.per_up,
            )?,
            down: self.build_argument_weight_table(
                ctx,
                match down_quant {
                    1 => ctx.qwen_moe_decode_q5k_table_arg_encoder.as_ref()?,
                    2 => ctx.qwen_moe_decode_q6k_table_arg_encoder.as_ref()?,
                    _ => return None,
                },
                down_sparse_raw,
                down_shared_raw,
                self.per_down,
            )?,
        })
    }

    pub(crate) fn debug_routes(&self) -> (Vec<u32>, Vec<f32>) {
        let ids = unsafe {
            std::slice::from_raw_parts(
                self.expert_ids_dev.contents().as_ptr() as *const u32,
                self.slots,
            )
        };
        let weights = unsafe {
            std::slice::from_raw_parts(
                self.route_weights_dev.contents().as_ptr() as *const f32,
                self.slots,
            )
        };
        (ids.to_vec(), weights.to_vec())
    }

    pub(crate) fn debug_normed(&self) -> Vec<f32> {
        readback(&self.normed_dev, self.n_embd)
    }

    pub(crate) fn debug_buffer_stats(&self) -> Vec<(&'static str, usize, f32, [f32; 4])> {
        fn stats(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> (usize, f32, [f32; 4]) {
            let values =
                unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len) };
            let mut nonfinite = 0usize;
            let mut max_abs = 0.0f32;
            for &value in values {
                if value.is_finite() {
                    max_abs = max_abs.max(value.abs());
                } else {
                    nonfinite += 1;
                }
            }
            let mut head = [0.0f32; 4];
            for (dst, &value) in head.iter_mut().zip(values.iter()) {
                *dst = value;
            }
            (nonfinite, max_abs, head)
        }

        let mut out = Vec::with_capacity(5);
        let (nonfinite, max_abs, head) = stats(&self.gate_dev, self.slots * self.n_ff);
        out.push(("gate", nonfinite, max_abs, head));
        let (nonfinite, max_abs, head) = stats(&self.up_dev, self.slots * self.n_ff);
        out.push(("up", nonfinite, max_abs, head));
        let (nonfinite, max_abs, head) = stats(&self.down_dev, self.slots * self.n_embd);
        out.push(("down", nonfinite, max_abs, head));
        let (nonfinite, max_abs, head) = stats(&self.moe_out_dev, self.n_embd);
        out.push(("moe_out", nonfinite, max_abs, head));
        let (nonfinite, max_abs, head) = stats(&self.normed_dev, self.n_embd);
        out.push(("normed", nonfinite, max_abs, head));
        out
    }
}

fn encode_qwen_moe_decode_route_shared(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    logits: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    shared_input_scale: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &ProtocolObject<dyn MTLBuffer>,
    route_weights: &ProtocolObject<dyn MTLBuffer>,
    n_expert_buf: &ProtocolObject<dyn MTLBuffer>,
    n_used_buf: &ProtocolObject<dyn MTLBuffer>,
    n_embd_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_expert_id_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    enc.setComputePipelineState(ctx.qwen_moe_decode_route_shared_pipeline());
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(input), 0, 1);
        enc.setBuffer_offset_atIndex(Some(shared_input_scale), 0, 2);
        enc.setBuffer_offset_atIndex(Some(expert_ids), 0, 3);
        enc.setBuffer_offset_atIndex(Some(route_weights), 0, 4);
        enc.setBuffer_offset_atIndex(Some(n_expert_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(n_used_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(n_embd_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(shared_expert_id_buf), 0, 8);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn qwen_moe_decode_simdgroups() -> usize {
    if std::env::var_os("RNB_METAL_QWEN35_MOE_DECODE_NSG4").is_some() {
        4
    } else {
        2
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_decode_q4k_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    sparse_w: &ProtocolObject<dyn MTLBuffer>,
    shared_w: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    per_expert_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_expert_id_buf: &ProtocolObject<dyn MTLBuffer>,
    sparse_off_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) {
    enc.setComputePipelineState(ctx.qwen_moe_decode_q4k_slots_pipeline());
    unsafe {
        enc.setBuffer_offset_atIndex(Some(sparse_w), 0, 0);
        enc.setBuffer_offset_atIndex(Some(shared_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(input), 0, 2);
        enc.setBuffer_offset_atIndex(Some(out), 0, 3);
        enc.setBuffer_offset_atIndex(Some(expert_ids), 0, 4);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(per_expert_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(shared_expert_id_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(sparse_off_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(shared_off_buf), 0, 10);
    }
    let simdgroups = qwen_moe_decode_simdgroups();
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2 * simdgroups),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: simdgroups,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_decode_q4k_pair_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    gate_shared_w: &ProtocolObject<dyn MTLBuffer>,
    up_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    up_shared_w: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    gate_out: &ProtocolObject<dyn MTLBuffer>,
    up_out: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_per_expert_buf: &ProtocolObject<dyn MTLBuffer>,
    up_per_expert_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_expert_id_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_off_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_shared_off_buf: &ProtocolObject<dyn MTLBuffer>,
    up_sparse_off_buf: &ProtocolObject<dyn MTLBuffer>,
    up_shared_off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) -> bool {
    let Some(pipeline) = ctx.qwen_moe_decode_q4k_pair_slots_pipeline.as_ref() else {
        return false;
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_sparse_w), 0, 0);
        enc.setBuffer_offset_atIndex(Some(gate_shared_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(up_sparse_w), 0, 2);
        enc.setBuffer_offset_atIndex(Some(up_shared_w), 0, 3);
        enc.setBuffer_offset_atIndex(Some(input), 0, 4);
        enc.setBuffer_offset_atIndex(Some(gate_out), 0, 5);
        enc.setBuffer_offset_atIndex(Some(up_out), 0, 6);
        enc.setBuffer_offset_atIndex(Some(expert_ids), 0, 7);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(gate_per_expert_buf), 0, 10);
        enc.setBuffer_offset_atIndex(Some(up_per_expert_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(shared_expert_id_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(gate_sparse_off_buf), 0, 13);
        enc.setBuffer_offset_atIndex(Some(gate_shared_off_buf), 0, 14);
        enc.setBuffer_offset_atIndex(Some(up_sparse_off_buf), 0, 15);
        enc.setBuffer_offset_atIndex(Some(up_shared_off_buf), 0, 16);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    true
}

fn set_qwen_moe_selected_weight_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    offsets: &[u32],
) {
    debug_assert!(!weights.is_empty());
    debug_assert_eq!(weights.len(), offsets.len());
    debug_assert!(weights.len() <= 9);
    for index in 0..9usize {
        let slot = index.min(weights.len() - 1);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&weights[slot]), offsets[slot] as usize, index);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn set_qwen_moe_selected_weight_buffers_from_ids(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    sparse_weight: &ProtocolObject<dyn MTLBuffer>,
    shared_weight: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &[u32],
    shared_expert_id: u32,
    sparse_byte_offset: u32,
    shared_byte_offset: u32,
    per_expert_bytes: u32,
) {
    debug_assert!(!expert_ids.is_empty());
    debug_assert!(expert_ids.len() <= 9);
    for index in 0..9usize {
        let slot = index.min(expert_ids.len() - 1);
        let expert = expert_ids[slot];
        let (weight, offset) = if expert == shared_expert_id {
            (shared_weight, shared_byte_offset as usize)
        } else {
            (
                sparse_weight,
                sparse_byte_offset as usize + expert as usize * per_expert_bytes as usize,
            )
        };
        unsafe {
            enc.setBuffer_offset_atIndex(Some(weight), offset, index);
        }
    }
}

fn encode_qwen_moe_decode_q4k_selected_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    offsets: &[u32],
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) {
    debug_assert!(slots > 0 && slots <= 9);
    debug_assert_eq!(weights.len(), slots);
    debug_assert_eq!(offsets.len(), slots);
    if std::env::var_os("RNB_METAL_QWEN35_MOE_SELECTED_Q4K_NSG2").is_some() {
        if let Some(pipeline) = ctx
            .qwen_moe_decode_q4k_selected_slots_nsg2_pipeline
            .as_ref()
        {
            enc.setComputePipelineState(pipeline);
            set_qwen_moe_selected_weight_buffers(enc, weights, offsets);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(input), 0, 9);
                enc.setBuffer_offset_atIndex(Some(out), 0, 10);
                enc.setBuffer_offset_atIndex(Some(n_buf), 0, 11);
                enc.setBuffer_offset_atIndex(Some(k_buf), 0, 12);
                enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 13);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n.div_ceil(4),
                    height: slots,
                    depth: 1,
                },
                MTLSize {
                    width: crate::compute::SIMD_WIDTH * 2,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }
    }
    enc.setComputePipelineState(ctx.qwen_moe_decode_q4k_selected_slots_coalesced_pipeline());
    set_qwen_moe_selected_weight_buffers(enc, weights, offsets);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 9);
        enc.setBuffer_offset_atIndex(Some(out), 0, 10);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 13);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_decode_q4k_selected_pair_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    gate_offsets: &[u32],
    up_weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    up_offsets: &[u32],
    input: &ProtocolObject<dyn MTLBuffer>,
    gate_out: &ProtocolObject<dyn MTLBuffer>,
    up_out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) -> bool {
    debug_assert!(slots > 0 && slots <= 9);
    debug_assert_eq!(gate_weights.len(), slots);
    debug_assert_eq!(gate_offsets.len(), slots);
    debug_assert_eq!(up_weights.len(), slots);
    debug_assert_eq!(up_offsets.len(), slots);
    let Some(pipeline) = ctx
        .qwen_moe_decode_q4k_selected_pair_slots_pipeline
        .as_ref()
    else {
        return false;
    };

    enc.setComputePipelineState(pipeline);
    for index in 0..9usize {
        let slot = index.min(slots - 1);
        unsafe {
            enc.setBuffer_offset_atIndex(
                Some(&gate_weights[slot]),
                gate_offsets[slot] as usize,
                index,
            );
            enc.setBuffer_offset_atIndex(
                Some(&up_weights[slot]),
                up_offsets[slot] as usize,
                index + 9,
            );
        }
    }
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 18);
        enc.setBuffer_offset_atIndex(Some(gate_out), 0, 19);
        enc.setBuffer_offset_atIndex(Some(up_out), 0, 20);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 21);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 22);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 23);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    true
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_decode_q4k_selected_id_offsets(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    sparse_weight: &ProtocolObject<dyn MTLBuffer>,
    shared_weight: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &[u32],
    shared_expert_id: u32,
    sparse_byte_offset: u32,
    shared_byte_offset: u32,
    per_expert_bytes: u32,
    n: usize,
    slots: usize,
) {
    debug_assert!(slots > 0 && slots <= 9);
    debug_assert_eq!(expert_ids.len(), slots);
    enc.setComputePipelineState(ctx.qwen_moe_decode_q4k_selected_slots_coalesced_pipeline());
    set_qwen_moe_selected_weight_buffers_from_ids(
        enc,
        sparse_weight,
        shared_weight,
        expert_ids,
        shared_expert_id,
        sparse_byte_offset,
        shared_byte_offset,
        per_expert_bytes,
    );
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 9);
        enc.setBuffer_offset_atIndex(Some(out), 0, 10);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 13);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_qwen_moe_decode_down_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    quant: u8,
    sparse_w: &ProtocolObject<dyn MTLBuffer>,
    shared_w: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    per_expert_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_expert_id_buf: &ProtocolObject<dyn MTLBuffer>,
    sparse_off_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_off_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) {
    let pipeline = match quant {
        0 => ctx.qwen_moe_decode_q4k_down_slots_pipeline(),
        1 => ctx.qwen_moe_decode_q5k_slots_pipeline(),
        2 => ctx.qwen_moe_decode_q6k_slots_pipeline(),
        other => panic!("unsupported Qwen MoE decode slots down quant {other}"),
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(sparse_w), 0, 0);
        enc.setBuffer_offset_atIndex(Some(shared_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(input), 0, 2);
        enc.setBuffer_offset_atIndex(Some(out), 0, 3);
        enc.setBuffer_offset_atIndex(Some(expert_ids), 0, 4);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(per_expert_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(shared_expert_id_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(sparse_off_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(shared_off_buf), 0, 10);
    }
    if quant != 1 {
        let simdgroups = qwen_moe_decode_simdgroups();
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(2 * simdgroups),
                height: slots,
                depth: 1,
            },
            MTLSize {
                width: crate::compute::SIMD_WIDTH,
                height: simdgroups,
                depth: 1,
            },
        );
        return;
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_decode_q6k_selected_id_offsets(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    sparse_weight: &ProtocolObject<dyn MTLBuffer>,
    shared_weight: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &[u32],
    shared_expert_id: u32,
    sparse_byte_offset: u32,
    shared_byte_offset: u32,
    per_expert_bytes: u32,
    n: usize,
    slots: usize,
) {
    debug_assert!(slots > 0 && slots <= 9);
    debug_assert_eq!(expert_ids.len(), slots);
    enc.setComputePipelineState(ctx.qwen_moe_decode_q6k_selected_slots_pipeline());
    set_qwen_moe_selected_weight_buffers_from_ids(
        enc,
        sparse_weight,
        shared_weight,
        expert_ids,
        shared_expert_id,
        sparse_byte_offset,
        shared_byte_offset,
        per_expert_bytes,
    );
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 9);
        enc.setBuffer_offset_atIndex(Some(out), 0, 10);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 13);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_decode_q5k_selected_id_offsets(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    sparse_weight: &ProtocolObject<dyn MTLBuffer>,
    shared_weight: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &[u32],
    shared_expert_id: u32,
    sparse_byte_offset: u32,
    shared_byte_offset: u32,
    per_expert_bytes: u32,
    n: usize,
    slots: usize,
) {
    debug_assert!(slots > 0 && slots <= 9);
    debug_assert_eq!(expert_ids.len(), slots);
    enc.setComputePipelineState(ctx.qwen_moe_decode_q5k_selected_slots_pipeline());
    set_qwen_moe_selected_weight_buffers_from_ids(
        enc,
        sparse_weight,
        shared_weight,
        expert_ids,
        shared_expert_id,
        sparse_byte_offset,
        shared_byte_offset,
        per_expert_bytes,
    );
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 9);
        enc.setBuffer_offset_atIndex(Some(out), 0, 10);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 13);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_qwen_moe_decode_q5k_selected_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    offsets: &[u32],
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) {
    debug_assert!(slots > 0 && slots <= 9);
    debug_assert_eq!(weights.len(), slots);
    debug_assert_eq!(offsets.len(), slots);
    enc.setComputePipelineState(ctx.qwen_moe_decode_q5k_selected_slots_pipeline());
    set_qwen_moe_selected_weight_buffers(enc, weights, offsets);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 9);
        enc.setBuffer_offset_atIndex(Some(out), 0, 10);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 13);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_qwen_moe_decode_q6k_selected_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    offsets: &[u32],
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) {
    debug_assert!(slots > 0 && slots <= 9);
    debug_assert_eq!(weights.len(), slots);
    debug_assert_eq!(offsets.len(), slots);
    enc.setComputePipelineState(ctx.qwen_moe_decode_q6k_selected_slots_pipeline());
    set_qwen_moe_selected_weight_buffers(enc, weights, offsets);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 9);
        enc.setBuffer_offset_atIndex(Some(out), 0, 10);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 13);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_glm_moe_decode_iq_selected_slots(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    offsets: &[u32],
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) {
    encode_glm_moe_decode_iq_selected_slots_at(
        pipeline, enc, weights, offsets, input, 0, out, n_buf, k_buf, slots_buf, n, slots,
    );
}

/// pm113: input 을 byte offset 으로 바인딩하는 변형 — prefill token-batch 에서
/// 큰 input 버퍼 `[seq][K]` 의 토큰별 구간을 가리킨다.
#[allow(clippy::too_many_arguments)]
fn encode_glm_moe_decode_iq_selected_slots_at(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    offsets: &[u32],
    input: &ProtocolObject<dyn MTLBuffer>,
    in_byte_offset: usize,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) {
    debug_assert!(slots > 0 && slots <= 9);
    debug_assert_eq!(weights.len(), slots);
    debug_assert_eq!(offsets.len(), slots);
    enc.setComputePipelineState(pipeline);
    set_qwen_moe_selected_weight_buffers(enc, weights, offsets);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), in_byte_offset, 9);
        enc.setBuffer_offset_atIndex(Some(out), 0, 10);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 11);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 12);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 13);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            // pm113: IQ 커널 nr0=4 (simdgroup 당 4행 × 4 simdgroups = tg 당 16행).
            width: n.div_ceil(16),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 4,
            depth: 1,
        },
    );
}

fn use_mtl_buffer_resource(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    usage: MTLResourceUsage,
) {
    enc.useResource_usage(ProtocolObject::<dyn MTLResource>::from_ref(buffer), usage);
}

fn use_qwen_moe_argument_table(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    table: &QwenMoeArgumentWeightTable,
) {
    debug_assert_eq!(table.weights.len(), table.weight_resources.len());
    use_mtl_buffer_resource(enc, &table.argument_buf, MTLResourceUsage::Read);
    if !table.weight_resources.is_empty() {
        unsafe {
            enc.useResources_count_usage(
                NonNull::new(table.weight_resources.as_ptr() as *mut _)
                    .expect("Qwen MoE table resource ptr"),
                table.weight_resources.len(),
                MTLResourceUsage::Read,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_decode_table_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    quant: u8,
    table: &QwenMoeArgumentWeightTable,
    input: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    expert_ids: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    slots: usize,
) {
    let pipeline = match quant {
        0 => ctx
            .qwen_moe_decode_q4k_table_slots_pipeline
            .as_ref()
            .expect("Qwen MoE Q4_K argument-table pipeline"),
        1 => ctx
            .qwen_moe_decode_q5k_table_slots_pipeline
            .as_ref()
            .expect("Qwen MoE Q5_K argument-table pipeline"),
        2 => ctx
            .qwen_moe_decode_q6k_table_slots_pipeline
            .as_ref()
            .expect("Qwen MoE Q6_K argument-table pipeline"),
        other => panic!("unsupported Qwen MoE argument-table quant {other}"),
    };
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&table.argument_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(input), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(expert_ids), 0, 3);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 6);
    }
    use_qwen_moe_argument_table(enc, table);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(2),
            height: slots,
            depth: 1,
        },
        MTLSize {
            width: crate::compute::SIMD_WIDTH,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_qwen_moe_decode_silu_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    total_buf: &ProtocolObject<dyn MTLBuffer>,
    total: usize,
) {
    enc.setComputePipelineState(ctx.qwen_moe_decode_silu_slots_pipeline());
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up), 0, 1);
        enc.setBuffer_offset_atIndex(Some(total_buf), 0, 2);
    }
    let tg_width = 256usize;
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: total.div_ceil(tg_width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_qwen_moe_decode_reduce_slots(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    down: &ProtocolObject<dyn MTLBuffer>,
    route_weights: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    encode_qwen_moe_decode_reduce_slots_at(
        ctx,
        enc,
        down,
        route_weights,
        0,
        out,
        0,
        n_buf,
        slots_buf,
        n,
    );
}

/// pm113: route/out 을 byte offset 으로 바인딩하는 변형 — prefill token-batch 에서
/// 큰 route `[seq][slots]` / out `[seq][n_embd]` 버퍼의 토큰별 구간을 가리킨다.
#[allow(clippy::too_many_arguments)]
fn encode_qwen_moe_decode_reduce_slots_at(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    down: &ProtocolObject<dyn MTLBuffer>,
    route_weights: &ProtocolObject<dyn MTLBuffer>,
    route_byte_offset: usize,
    out: &ProtocolObject<dyn MTLBuffer>,
    out_byte_offset: usize,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    slots_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) {
    enc.setComputePipelineState(ctx.qwen_moe_decode_reduce_slots_pipeline());
    unsafe {
        enc.setBuffer_offset_atIndex(Some(down), 0, 0);
        enc.setBuffer_offset_atIndex(Some(route_weights), route_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(out), out_byte_offset, 2);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(slots_buf), 0, 4);
    }
    let tg_width = 256usize;
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(tg_width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_chain_encode_router_logits(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeDecodeChainCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    router_w_buf: &ProtocolObject<dyn MTLBuffer>,
    router_off_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_off: u32,
    up_sparse_off: u32,
    down_sparse_off: u32,
    gate_shared_off: u32,
    up_shared_off: u32,
    down_shared_off: u32,
) {
    carrier.upload_offsets(
        gate_sparse_off,
        up_sparse_off,
        down_sparse_off,
        gate_shared_off,
        up_shared_off,
        down_shared_off,
    );

    encode_rms_norm(
        ctx,
        enc,
        hidden_dev,
        ffn_norm_w_buf,
        &carrier.normed_dev,
        &carrier.n_embd_buf,
        &carrier.eps_buf,
    );
    crate::compute::chain_barrier(ctx, enc);

    crate::compute::encode_gemv_f32_router_simd(
        ctx,
        enc,
        router_w_buf,
        &carrier.normed_dev,
        &carrier.router_logits_dev,
        &carrier.n_expert_buf,
        &carrier.n_embd_buf,
        router_off_buf,
        carrier.n_expert,
    );
    crate::compute::chain_barrier(ctx, enc);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_chain_encode_route(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeDecodeChainCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    router_w_buf: &ProtocolObject<dyn MTLBuffer>,
    router_off_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_input_scale_buf: &ProtocolObject<dyn MTLBuffer>,
    _gate_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_off: u32,
    _up_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    up_sparse_off: u32,
    _down_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    down_sparse_off: u32,
    _gate_shared_w: &ProtocolObject<dyn MTLBuffer>,
    gate_shared_off: u32,
    _up_shared_w: &ProtocolObject<dyn MTLBuffer>,
    up_shared_off: u32,
    _down_shared_w: &ProtocolObject<dyn MTLBuffer>,
    down_shared_off: u32,
    down_quant: u8,
) {
    debug_assert_eq!(carrier.slots, carrier.n_used + 1);
    debug_assert!(
        matches!(down_quant, 0 | 1 | 2),
        "Qwen MoE decode chain supports Q4_K/Q5_K/Q6_K down"
    );
    qwen_moe_decode_chain_encode_router_logits(
        ctx,
        enc,
        carrier,
        hidden_dev,
        ffn_norm_w_buf,
        router_w_buf,
        router_off_buf,
        gate_sparse_off,
        up_sparse_off,
        down_sparse_off,
        gate_shared_off,
        up_shared_off,
        down_shared_off,
    );

    encode_qwen_moe_decode_route_shared(
        ctx,
        enc,
        &carrier.router_logits_dev,
        &carrier.normed_dev,
        shared_input_scale_buf,
        &carrier.expert_ids_dev,
        &carrier.route_weights_dev,
        &carrier.n_expert_buf,
        &carrier.n_used_buf,
        &carrier.n_embd_buf,
        &carrier.shared_expert_id_buf,
    );
    crate::compute::chain_barrier(ctx, enc);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_chain_encode_after_route(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeDecodeChainCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    up_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    down_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    gate_shared_w: &ProtocolObject<dyn MTLBuffer>,
    up_shared_w: &ProtocolObject<dyn MTLBuffer>,
    down_shared_w: &ProtocolObject<dyn MTLBuffer>,
    down_quant: u8,
) {
    debug_assert_eq!(carrier.slots, carrier.n_used + 1);
    debug_assert!(
        matches!(down_quant, 0 | 1 | 2),
        "Qwen MoE decode chain supports Q4_K/Q5_K/Q6_K down"
    );
    let used_gate_up_pair =
        std::env::var_os("RNB_METAL_QWEN35_MOE_GATE_UP_PAIR").is_some_and(|_| {
            encode_qwen_moe_decode_q4k_pair_slots(
                ctx,
                enc,
                gate_sparse_w,
                gate_shared_w,
                up_sparse_w,
                up_shared_w,
                &carrier.normed_dev,
                &carrier.gate_dev,
                &carrier.up_dev,
                &carrier.expert_ids_dev,
                &carrier.n_ff_buf,
                &carrier.n_embd_buf,
                &carrier.per_gate_buf,
                &carrier.per_up_buf,
                &carrier.shared_expert_id_buf,
                &carrier.gate_sparse_off_buf,
                &carrier.gate_shared_off_buf,
                &carrier.up_sparse_off_buf,
                &carrier.up_shared_off_buf,
                carrier.n_ff,
                carrier.slots,
            )
        });
    if !used_gate_up_pair {
        encode_qwen_moe_decode_q4k_slots(
            ctx,
            enc,
            gate_sparse_w,
            gate_shared_w,
            &carrier.normed_dev,
            &carrier.gate_dev,
            &carrier.expert_ids_dev,
            &carrier.n_ff_buf,
            &carrier.n_embd_buf,
            &carrier.per_gate_buf,
            &carrier.shared_expert_id_buf,
            &carrier.gate_sparse_off_buf,
            &carrier.gate_shared_off_buf,
            carrier.n_ff,
            carrier.slots,
        );
        encode_qwen_moe_decode_q4k_slots(
            ctx,
            enc,
            up_sparse_w,
            up_shared_w,
            &carrier.normed_dev,
            &carrier.up_dev,
            &carrier.expert_ids_dev,
            &carrier.n_ff_buf,
            &carrier.n_embd_buf,
            &carrier.per_up_buf,
            &carrier.shared_expert_id_buf,
            &carrier.up_sparse_off_buf,
            &carrier.up_shared_off_buf,
            carrier.n_ff,
            carrier.slots,
        );
    }
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_silu_slots(
        ctx,
        enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.total_ff_buf,
        carrier.slots * carrier.n_ff,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_down_slots(
        ctx,
        enc,
        down_quant,
        down_sparse_w,
        down_shared_w,
        &carrier.gate_dev,
        &carrier.down_dev,
        &carrier.expert_ids_dev,
        &carrier.n_embd_buf,
        &carrier.n_ff_buf,
        &carrier.per_down_buf,
        &carrier.shared_expert_id_buf,
        &carrier.down_sparse_off_buf,
        &carrier.down_shared_off_buf,
        carrier.n_embd,
        carrier.slots,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_reduce_slots(
        ctx,
        enc,
        &carrier.down_dev,
        &carrier.route_weights_dev,
        &carrier.moe_out_dev,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_embd,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_residual_add(
        ctx,
        enc,
        hidden_dev,
        &carrier.moe_out_dev,
        &carrier.n_embd_buf,
        carrier.n_embd,
    );
    crate::compute::chain_barrier(ctx, enc);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_chain_encode_after_route_bound_offsets(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeDecodeChainCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_off: u32,
    up_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    up_sparse_off: u32,
    down_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    down_sparse_off: u32,
    gate_shared_w: &ProtocolObject<dyn MTLBuffer>,
    gate_shared_off: u32,
    up_shared_w: &ProtocolObject<dyn MTLBuffer>,
    up_shared_off: u32,
    down_shared_w: &ProtocolObject<dyn MTLBuffer>,
    down_shared_off: u32,
    expert_ids: &[u32],
    route_weights: &[f32],
    down_quant: u8,
) -> bool {
    debug_assert_eq!(carrier.slots, carrier.n_used + 1);
    debug_assert_eq!(expert_ids.len(), carrier.slots);
    debug_assert_eq!(route_weights.len(), carrier.slots);
    if !matches!(down_quant, 1 | 2) || carrier.slots > 9 {
        return false;
    }
    carrier.upload_routes(expert_ids, route_weights);

    let shared_expert_id = carrier.n_expert as u32;
    encode_qwen_moe_decode_q4k_selected_id_offsets(
        ctx,
        enc,
        gate_sparse_w,
        gate_shared_w,
        &carrier.normed_dev,
        &carrier.gate_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        expert_ids,
        shared_expert_id,
        gate_sparse_off,
        gate_shared_off,
        carrier.per_gate as u32,
        carrier.n_ff,
        carrier.slots,
    );
    encode_qwen_moe_decode_q4k_selected_id_offsets(
        ctx,
        enc,
        up_sparse_w,
        up_shared_w,
        &carrier.normed_dev,
        &carrier.up_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        expert_ids,
        shared_expert_id,
        up_sparse_off,
        up_shared_off,
        carrier.per_up as u32,
        carrier.n_ff,
        carrier.slots,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_silu_slots(
        ctx,
        enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.total_ff_buf,
        carrier.slots * carrier.n_ff,
    );
    crate::compute::chain_barrier(ctx, enc);

    match down_quant {
        1 => encode_qwen_moe_decode_q5k_selected_id_offsets(
            ctx,
            enc,
            down_sparse_w,
            down_shared_w,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.n_embd_buf,
            &carrier.n_ff_buf,
            &carrier.slots_buf,
            expert_ids,
            shared_expert_id,
            down_sparse_off,
            down_shared_off,
            carrier.per_down as u32,
            carrier.n_embd,
            carrier.slots,
        ),
        2 => encode_qwen_moe_decode_q6k_selected_id_offsets(
            ctx,
            enc,
            down_sparse_w,
            down_shared_w,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.n_embd_buf,
            &carrier.n_ff_buf,
            &carrier.slots_buf,
            expert_ids,
            shared_expert_id,
            down_sparse_off,
            down_shared_off,
            carrier.per_down as u32,
            carrier.n_embd,
            carrier.slots,
        ),
        _ => unreachable!("down quant validated above"),
    }
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_reduce_slots(
        ctx,
        enc,
        &carrier.down_dev,
        &carrier.route_weights_dev,
        &carrier.moe_out_dev,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_embd,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_residual_add(
        ctx,
        enc,
        hidden_dev,
        &carrier.moe_out_dev,
        &carrier.n_embd_buf,
        carrier.n_embd,
    );
    crate::compute::chain_barrier(ctx, enc);
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_chain_encode_after_route_selected_buffers(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeDecodeChainCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    gate_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    gate_off: &[u32],
    up_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    up_off: &[u32],
    down_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    down_off: &[u32],
    expert_ids: &[u32],
    route_weights: &[f32],
    down_quant: u8,
) -> bool {
    debug_assert_eq!(carrier.slots, carrier.n_used + 1);
    debug_assert_eq!(expert_ids.len(), carrier.slots);
    debug_assert_eq!(route_weights.len(), carrier.slots);
    debug_assert_eq!(gate_w.len(), carrier.slots);
    debug_assert_eq!(gate_off.len(), carrier.slots);
    debug_assert_eq!(up_w.len(), carrier.slots);
    debug_assert_eq!(up_off.len(), carrier.slots);
    debug_assert_eq!(down_w.len(), carrier.slots);
    debug_assert_eq!(down_off.len(), carrier.slots);
    if !matches!(down_quant, 1 | 2) || carrier.slots > 9 {
        return false;
    }
    carrier.upload_routes(expert_ids, route_weights);

    encode_qwen_moe_decode_q4k_selected_slots(
        ctx,
        enc,
        gate_w,
        gate_off,
        &carrier.normed_dev,
        &carrier.gate_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_ff,
        carrier.slots,
    );
    encode_qwen_moe_decode_q4k_selected_slots(
        ctx,
        enc,
        up_w,
        up_off,
        &carrier.normed_dev,
        &carrier.up_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_ff,
        carrier.slots,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_silu_slots(
        ctx,
        enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.total_ff_buf,
        carrier.slots * carrier.n_ff,
    );
    crate::compute::chain_barrier(ctx, enc);

    match down_quant {
        1 => encode_qwen_moe_decode_q5k_selected_slots(
            ctx,
            enc,
            down_w,
            down_off,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.n_embd_buf,
            &carrier.n_ff_buf,
            &carrier.slots_buf,
            carrier.n_embd,
            carrier.slots,
        ),
        2 => encode_qwen_moe_decode_q6k_selected_slots(
            ctx,
            enc,
            down_w,
            down_off,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.n_embd_buf,
            &carrier.n_ff_buf,
            &carrier.slots_buf,
            carrier.n_embd,
            carrier.slots,
        ),
        _ => unreachable!("down quant validated above"),
    }
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_reduce_slots(
        ctx,
        enc,
        &carrier.down_dev,
        &carrier.route_weights_dev,
        &carrier.moe_out_dev,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_embd,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_residual_add(
        ctx,
        enc,
        hidden_dev,
        &carrier.moe_out_dev,
        &carrier.n_embd_buf,
        carrier.n_embd,
    );
    crate::compute::chain_barrier(ctx, enc);
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_chain_encode_after_route_argument_table(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeDecodeChainCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_raw: &[u8],
    up_sparse_raw: &[u8],
    down_sparse_raw: &[u8],
    gate_shared_raw: &[u8],
    up_shared_raw: &[u8],
    down_shared_raw: &[u8],
    down_quant: u8,
) -> bool {
    debug_assert_eq!(carrier.slots, carrier.n_used + 1);
    if !matches!(down_quant, 1 | 2) || carrier.n_expert + 1 > QWEN_MOE_WEIGHT_TABLE_CAP {
        return false;
    }
    let down_pipeline_available = match down_quant {
        1 => ctx.qwen_moe_decode_q5k_table_slots_pipeline.is_some(),
        2 => ctx.qwen_moe_decode_q6k_table_slots_pipeline.is_some(),
        _ => unreachable!("down quant validated above"),
    };
    if ctx.qwen_moe_decode_q4k_table_slots_pipeline.is_none() || !down_pipeline_available {
        return false;
    }

    let key = QwenMoeDecodeTableKey {
        gate_sparse: (gate_sparse_raw.as_ptr() as usize, gate_sparse_raw.len()),
        up_sparse: (up_sparse_raw.as_ptr() as usize, up_sparse_raw.len()),
        down_sparse: (down_sparse_raw.as_ptr() as usize, down_sparse_raw.len()),
        gate_shared: (gate_shared_raw.as_ptr() as usize, gate_shared_raw.len()),
        up_shared: (up_shared_raw.as_ptr() as usize, up_shared_raw.len()),
        down_shared: (down_shared_raw.as_ptr() as usize, down_shared_raw.len()),
    };
    if !carrier.table_cache.borrow().contains_key(&key) {
        let Some(table_set) = carrier.build_table_set(
            ctx,
            down_quant,
            gate_sparse_raw,
            up_sparse_raw,
            down_sparse_raw,
            gate_shared_raw,
            up_shared_raw,
            down_shared_raw,
        ) else {
            return false;
        };
        carrier.table_cache.borrow_mut().insert(key, table_set);
    }
    let table_cache = carrier.table_cache.borrow();
    let table_set = table_cache
        .get(&key)
        .expect("Qwen MoE argument table cached");

    encode_qwen_moe_decode_table_slots(
        ctx,
        enc,
        0,
        &table_set.gate,
        &carrier.normed_dev,
        &carrier.gate_dev,
        &carrier.expert_ids_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_ff,
        carrier.slots,
    );
    encode_qwen_moe_decode_table_slots(
        ctx,
        enc,
        0,
        &table_set.up,
        &carrier.normed_dev,
        &carrier.up_dev,
        &carrier.expert_ids_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_ff,
        carrier.slots,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_silu_slots(
        ctx,
        enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.total_ff_buf,
        carrier.slots * carrier.n_ff,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_table_slots(
        ctx,
        enc,
        down_quant,
        &table_set.down,
        &carrier.gate_dev,
        &carrier.down_dev,
        &carrier.expert_ids_dev,
        &carrier.n_embd_buf,
        &carrier.n_ff_buf,
        &carrier.slots_buf,
        carrier.n_embd,
        carrier.slots,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_qwen_moe_decode_reduce_slots(
        ctx,
        enc,
        &carrier.down_dev,
        &carrier.route_weights_dev,
        &carrier.moe_out_dev,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_embd,
    );
    crate::compute::chain_barrier(ctx, enc);

    encode_residual_add(
        ctx,
        enc,
        hidden_dev,
        &carrier.moe_out_dev,
        &carrier.n_embd_buf,
        carrier.n_embd,
    );
    crate::compute::chain_barrier(ctx, enc);
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_chain_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeDecodeChainCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    ffn_norm_w_buf: &ProtocolObject<dyn MTLBuffer>,
    router_w_buf: &ProtocolObject<dyn MTLBuffer>,
    router_off_buf: &ProtocolObject<dyn MTLBuffer>,
    shared_input_scale_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_off: u32,
    up_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    up_sparse_off: u32,
    down_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    down_sparse_off: u32,
    gate_shared_w: &ProtocolObject<dyn MTLBuffer>,
    gate_shared_off: u32,
    up_shared_w: &ProtocolObject<dyn MTLBuffer>,
    up_shared_off: u32,
    down_shared_w: &ProtocolObject<dyn MTLBuffer>,
    down_shared_off: u32,
    down_quant: u8,
) {
    qwen_moe_decode_chain_encode_route(
        ctx,
        enc,
        carrier,
        hidden_dev,
        ffn_norm_w_buf,
        router_w_buf,
        router_off_buf,
        shared_input_scale_buf,
        gate_sparse_w,
        gate_sparse_off,
        up_sparse_w,
        up_sparse_off,
        down_sparse_w,
        down_sparse_off,
        gate_shared_w,
        gate_shared_off,
        up_shared_w,
        up_shared_off,
        down_shared_w,
        down_shared_off,
        down_quant,
    );
    qwen_moe_decode_chain_encode_after_route(
        ctx,
        enc,
        carrier,
        hidden_dev,
        gate_sparse_w,
        up_sparse_w,
        down_sparse_w,
        gate_shared_w,
        up_shared_w,
        down_shared_w,
        down_quant,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_id_dispatch(
    ctx: &MetalContext,
    carrier: &QwenMoeDecodeIdCarrier,
    gate_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    gate_sparse_off: u32,
    up_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    up_sparse_off: u32,
    down_sparse_w: &ProtocolObject<dyn MTLBuffer>,
    down_sparse_off: u32,
    gate_shared_w: &ProtocolObject<dyn MTLBuffer>,
    gate_shared_off: u32,
    up_shared_w: &ProtocolObject<dyn MTLBuffer>,
    up_shared_off: u32,
    down_shared_w: &ProtocolObject<dyn MTLBuffer>,
    down_shared_off: u32,
    input: &[f32],
    expert_ids: &[u32],
    route_weights: &[f32],
    down_quant: u8,
) -> Vec<f32> {
    assert_eq!(expert_ids.len(), carrier.slots);
    assert_eq!(route_weights.len(), carrier.slots);
    assert!(
        matches!(down_quant, 0 | 1 | 2),
        "Qwen MoE decode id path supports Q4_K/Q5_K/Q6_K down"
    );
    carrier.upload_input(input);
    carrier.upload_routes(expert_ids, route_weights);
    carrier.upload_offsets(
        gate_sparse_off,
        up_sparse_off,
        down_sparse_off,
        gate_shared_off,
        up_shared_off,
        down_shared_off,
    );

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = crate::compute::chain_compute_encoder(ctx, &cmd);

    encode_qwen_moe_decode_q4k_slots(
        ctx,
        &enc,
        gate_sparse_w,
        gate_shared_w,
        &carrier.input_dev,
        &carrier.gate_dev,
        &carrier.expert_ids_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.per_gate_buf,
        &carrier.shared_expert_id_buf,
        &carrier.gate_sparse_off_buf,
        &carrier.gate_shared_off_buf,
        carrier.n_ff,
        carrier.slots,
    );
    encode_qwen_moe_decode_q4k_slots(
        ctx,
        &enc,
        up_sparse_w,
        up_shared_w,
        &carrier.input_dev,
        &carrier.up_dev,
        &carrier.expert_ids_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.per_up_buf,
        &carrier.shared_expert_id_buf,
        &carrier.up_sparse_off_buf,
        &carrier.up_shared_off_buf,
        carrier.n_ff,
        carrier.slots,
    );
    crate::compute::chain_barrier(ctx, &enc);
    encode_qwen_moe_decode_silu_slots(
        ctx,
        &enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.total_ff_buf,
        carrier.slots * carrier.n_ff,
    );
    crate::compute::chain_barrier(ctx, &enc);
    encode_qwen_moe_decode_down_slots(
        ctx,
        &enc,
        down_quant,
        down_sparse_w,
        down_shared_w,
        &carrier.gate_dev,
        &carrier.down_dev,
        &carrier.expert_ids_dev,
        &carrier.n_embd_buf,
        &carrier.n_ff_buf,
        &carrier.per_down_buf,
        &carrier.shared_expert_id_buf,
        &carrier.down_sparse_off_buf,
        &carrier.down_shared_off_buf,
        carrier.n_embd,
        carrier.slots,
    );
    crate::compute::chain_barrier(ctx, &enc);
    encode_qwen_moe_decode_reduce_slots(
        ctx,
        &enc,
        &carrier.down_dev,
        &carrier.route_weights_dev,
        &carrier.out_dev,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_embd,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    readback(&carrier.out_dev, carrier.n_embd)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_decode_dispatch(
    ctx: &MetalContext,
    carrier: &QwenMoeDecodeCarrier,
    input: &[f32],
    gate_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    gate_off: &[u32],
    up_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    up_off: &[u32],
    down_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    down_off: &[u32],
    route_weights: &[f32],
    down_quant: u8,
) -> Vec<f32> {
    let slots = route_weights.len();
    debug_assert_eq!(slots, carrier.slots);
    debug_assert_eq!(gate_w.len(), slots);
    debug_assert_eq!(gate_off.len(), slots);
    debug_assert_eq!(up_w.len(), slots);
    debug_assert_eq!(up_off.len(), slots);
    debug_assert_eq!(down_w.len(), slots);
    debug_assert_eq!(down_off.len(), slots);

    carrier.upload_input(input);
    carrier.upload_offsets(gate_off, up_off, down_off);
    carrier.upload_route_weights(route_weights);
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    if slots <= 9 && down_quant == 2 {
        let used_selected_gate_up_pair =
            std::env::var_os("RNB_METAL_QWEN35_MOE_SELECTED_GATE_UP_PAIR").is_some_and(|_| {
                encode_qwen_moe_decode_q4k_selected_pair_slots(
                    ctx,
                    &enc,
                    gate_w,
                    gate_off,
                    up_w,
                    up_off,
                    &carrier.input_dev,
                    &carrier.gate_dev,
                    &carrier.up_dev,
                    &carrier.n_ff_buf,
                    &carrier.n_embd_buf,
                    &carrier.slots_buf,
                    carrier.n_ff,
                    slots,
                )
            });
        if !used_selected_gate_up_pair {
            encode_qwen_moe_decode_q4k_selected_slots(
                ctx,
                &enc,
                gate_w,
                gate_off,
                &carrier.input_dev,
                &carrier.gate_dev,
                &carrier.n_ff_buf,
                &carrier.n_embd_buf,
                &carrier.slots_buf,
                carrier.n_ff,
                slots,
            );
            encode_qwen_moe_decode_q4k_selected_slots(
                ctx,
                &enc,
                up_w,
                up_off,
                &carrier.input_dev,
                &carrier.up_dev,
                &carrier.n_ff_buf,
                &carrier.n_embd_buf,
                &carrier.slots_buf,
                carrier.n_ff,
                slots,
            );
        }
        encode_silu_mul(
            ctx,
            &enc,
            &carrier.gate_dev,
            &carrier.up_dev,
            &carrier.total_ff_buf,
            carrier.slots * carrier.n_ff,
        );
        encode_qwen_moe_decode_q6k_selected_slots(
            ctx,
            &enc,
            down_w,
            down_off,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.n_embd_buf,
            &carrier.n_ff_buf,
            &carrier.slots_buf,
            carrier.n_embd,
            slots,
        );
    } else {
        let f32_bytes = std::mem::size_of::<f32>();
        for slot in 0..slots {
            let ff_offset = slot * carrier.n_ff * f32_bytes;
            crate::compute::encode_gemv_q4k_auto_offset(
                ctx,
                &enc,
                &gate_w[slot],
                &carrier.input_dev,
                &carrier.gate_dev,
                &carrier.n_ff_buf,
                &carrier.n_embd_buf,
                &carrier.gate_off_buf[slot],
                0,
                ff_offset,
                carrier.n_ff,
            );
            crate::compute::encode_gemv_q4k_auto_offset(
                ctx,
                &enc,
                &up_w[slot],
                &carrier.input_dev,
                &carrier.up_dev,
                &carrier.n_ff_buf,
                &carrier.n_embd_buf,
                &carrier.up_off_buf[slot],
                0,
                ff_offset,
                carrier.n_ff,
            );
        }

        encode_silu_mul(
            ctx,
            &enc,
            &carrier.gate_dev,
            &carrier.up_dev,
            &carrier.total_ff_buf,
            carrier.slots * carrier.n_ff,
        );

        for slot in 0..slots {
            let ff_offset = slot * carrier.n_ff * f32_bytes;
            let embd_offset = slot * carrier.n_embd * f32_bytes;
            match down_quant {
                0 => crate::compute::encode_gemv_q4k_auto_offset(
                    ctx,
                    &enc,
                    &down_w[slot],
                    &carrier.gate_dev,
                    &carrier.down_dev,
                    &carrier.n_embd_buf,
                    &carrier.n_ff_buf,
                    &carrier.down_off_buf[slot],
                    ff_offset,
                    embd_offset,
                    carrier.n_embd,
                ),
                1 => crate::compute::encode_gemv_q5k_auto_offset(
                    ctx,
                    &enc,
                    &down_w[slot],
                    &carrier.gate_dev,
                    &carrier.down_dev,
                    &carrier.n_embd_buf,
                    &carrier.n_ff_buf,
                    &carrier.down_off_buf[slot],
                    ff_offset,
                    embd_offset,
                    carrier.n_embd,
                ),
                2 => crate::compute::encode_gemv_q6k_auto_offset(
                    ctx,
                    &enc,
                    &down_w[slot],
                    &carrier.gate_dev,
                    &carrier.down_dev,
                    &carrier.n_embd_buf,
                    &carrier.n_ff_buf,
                    &carrier.down_off_buf[slot],
                    ff_offset,
                    embd_offset,
                    carrier.n_embd,
                ),
                other => panic!("unsupported Qwen MoE decode down quant code {other}"),
            }
        }
    }

    encode_qwen_moe_decode_reduce_slots(
        ctx,
        &enc,
        &carrier.down_dev,
        &carrier.route_weights_dev,
        &carrier.out_dev,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_embd,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    readback(&carrier.out_dev, carrier.n_embd)
}

/// pm113: GLM MoE prefill token-batch. 토큰별 stage 시퀀스(gate/up sparse+shared →
/// silu → down sparse+shared → reduce)를 **단일 command buffer** 에 encode 해
/// per-token commit/wait/wiring 오버헤드를 없앤다. carrier scratch 는 토큰 간
/// 공유 — Metal hazard tracking 이 토큰 순서를 직렬화한다 (decode 와 동일 산술).
///
/// slot 배열들은 `[seq][slots]` flatten (slots = sparse_slots + 1, 마지막이 shared).
/// shared weight 는 토큰 불변이므로 carrier 의 shared off_buf 는 1회 업로드.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_moe_prefill_iq_batch_dispatch(
    ctx: &MetalContext,
    carrier: &QwenMoeDecodeCarrier,
    input_all: &[f32],
    seq_len: usize,
    sparse_slots: usize,
    gate_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    gate_off: &[u32],
    up_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    up_off: &[u32],
    down_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    down_off: &[u32],
    route_weights_all: &[f32],
    select: crate::GlmMoeQuantSelect,
) -> Vec<f32> {
    let slots = sparse_slots + 1;
    debug_assert_eq!(slots, carrier.slots);
    debug_assert!(sparse_slots > 0 && sparse_slots <= 8);
    debug_assert_eq!(gate_w.len(), seq_len * slots);
    debug_assert_eq!(gate_off.len(), seq_len * slots);
    debug_assert_eq!(up_w.len(), seq_len * slots);
    debug_assert_eq!(up_off.len(), seq_len * slots);
    debug_assert_eq!(down_w.len(), seq_len * slots);
    debug_assert_eq!(down_off.len(), seq_len * slots);
    debug_assert_eq!(route_weights_all.len(), seq_len * slots);
    debug_assert_eq!(input_all.len(), seq_len * carrier.n_embd);

    let n_ff = carrier.n_ff;
    let n_embd = carrier.n_embd;
    let f32_bytes = std::mem::size_of::<f32>();
    let shared = MTLResourceOptions::StorageModeShared;
    let input_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(input_all.as_ptr() as *mut std::ffi::c_void)
            .expect("prefill input ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(ptr, std::mem::size_of_val(input_all), shared)
            .expect("Metal: GLM prefill input buffer")
    };
    let route_buf: Retained<ProtocolObject<dyn MTLBuffer>> = unsafe {
        let ptr = NonNull::new(route_weights_all.as_ptr() as *mut std::ffi::c_void)
            .expect("prefill route ptr is null");
        ctx.device
            .newBufferWithBytes_length_options(
                ptr,
                std::mem::size_of_val(route_weights_all),
                shared,
            )
            .expect("Metal: GLM prefill route buffer")
    };
    let out_buf: Retained<ProtocolObject<dyn MTLBuffer>> = ctx
        .device
        .newBufferWithLength_options(seq_len * n_embd * f32_bytes, shared)
        .expect("Metal: GLM prefill out buffer");

    // shared slot 의 weight byte offset 은 토큰 불변 — carrier off_buf 에 1회 업로드.
    carrier.upload_offsets(&gate_off[..slots], &up_off[..slots], &down_off[..slots]);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    let gate_up_pipeline = if select.gate_up_iq2s {
        ctx.glm_moe_decode_iq2s_selected_slots_pipeline()
    } else {
        ctx.glm_moe_decode_iq2xxs_selected_slots_pipeline()
    };
    let down_pipeline = if select.down_iq4xs {
        ctx.glm_moe_decode_iq4xs_selected_slots_pipeline()
    } else {
        ctx.glm_moe_decode_iq3xxs_selected_slots_pipeline()
    };
    let encode_shared_gate_up = if select.shared_gate_up_q6k {
        crate::compute::encode_gemv_q6k_auto_offset
    } else {
        crate::compute::encode_gemv_q5k_auto_offset
    };
    let shared_slot = sparse_slots;
    let shared_ff_offset = shared_slot * n_ff * f32_bytes;
    let shared_embd_offset = shared_slot * n_embd * f32_bytes;
    let zero_off_buf = u32_buf(ctx, 0);

    for token in 0..seq_len {
        let base = token * slots;
        let in_off = token * n_embd * f32_bytes;
        encode_glm_moe_decode_iq_selected_slots_at(
            gate_up_pipeline,
            &enc,
            &gate_w[base..base + sparse_slots],
            &gate_off[base..base + sparse_slots],
            &input_buf,
            in_off,
            &carrier.gate_dev,
            &carrier.n_ff_buf,
            &carrier.n_embd_buf,
            &carrier.slots_buf,
            n_ff,
            sparse_slots,
        );
        encode_glm_moe_decode_iq_selected_slots_at(
            gate_up_pipeline,
            &enc,
            &up_w[base..base + sparse_slots],
            &up_off[base..base + sparse_slots],
            &input_buf,
            in_off,
            &carrier.up_dev,
            &carrier.n_ff_buf,
            &carrier.n_embd_buf,
            &carrier.slots_buf,
            n_ff,
            sparse_slots,
        );
        encode_shared_gate_up(
            ctx,
            &enc,
            &gate_w[base + shared_slot],
            &input_buf,
            &carrier.gate_dev,
            &carrier.n_ff_buf,
            &carrier.n_embd_buf,
            &carrier.gate_off_buf[shared_slot],
            in_off,
            shared_ff_offset,
            n_ff,
        );
        encode_shared_gate_up(
            ctx,
            &enc,
            &up_w[base + shared_slot],
            &input_buf,
            &carrier.up_dev,
            &carrier.n_ff_buf,
            &carrier.n_embd_buf,
            &carrier.up_off_buf[shared_slot],
            in_off,
            shared_ff_offset,
            n_ff,
        );
        encode_silu_mul(
            ctx,
            &enc,
            &carrier.gate_dev,
            &carrier.up_dev,
            &carrier.total_ff_buf,
            carrier.slots * n_ff,
        );
        encode_glm_moe_decode_iq_selected_slots_at(
            down_pipeline,
            &enc,
            &down_w[base..base + sparse_slots],
            &down_off[base..base + sparse_slots],
            &carrier.gate_dev,
            0,
            &carrier.down_dev,
            &carrier.n_embd_buf,
            &carrier.n_ff_buf,
            &carrier.slots_buf,
            n_embd,
            sparse_slots,
        );
        if select.shared_down_q8_0 {
            crate::compute::encode_gemv_q8_0_at(
                ctx,
                &enc,
                &down_w[base + shared_slot],
                down_off[base + shared_slot] as usize,
                &carrier.gate_dev,
                shared_ff_offset,
                &carrier.down_dev,
                shared_embd_offset,
                &carrier.n_embd_buf,
                &carrier.n_ff_buf,
                &zero_off_buf,
                n_embd,
            );
        } else {
            crate::compute::encode_gemv_q6k_auto_offset(
                ctx,
                &enc,
                &down_w[base + shared_slot],
                &carrier.gate_dev,
                &carrier.down_dev,
                &carrier.n_embd_buf,
                &carrier.n_ff_buf,
                &carrier.down_off_buf[shared_slot],
                shared_ff_offset,
                shared_embd_offset,
                n_embd,
            );
        }
        encode_qwen_moe_decode_reduce_slots_at(
            ctx,
            &enc,
            &carrier.down_dev,
            &route_buf,
            base * f32_bytes,
            &out_buf,
            token * n_embd * f32_bytes,
            &carrier.n_embd_buf,
            &carrier.slots_buf,
            n_embd,
        );
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    readback(&out_buf, seq_len * n_embd)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_moe_decode_iq2xxs_iq3xxs_dispatch(
    ctx: &MetalContext,
    carrier: &QwenMoeDecodeCarrier,
    input: &[f32],
    sparse_slots: usize,
    gate_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    gate_off: &[u32],
    up_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    up_off: &[u32],
    down_w: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    down_off: &[u32],
    route_weights: &[f32],
    select: crate::GlmMoeQuantSelect,
) -> Vec<f32> {
    let slots = route_weights.len();
    debug_assert_eq!(slots, carrier.slots);
    debug_assert_eq!(sparse_slots + 1, slots);
    debug_assert!(sparse_slots > 0 && sparse_slots <= 8);
    debug_assert_eq!(gate_w.len(), slots);
    debug_assert_eq!(gate_off.len(), slots);
    debug_assert_eq!(up_w.len(), slots);
    debug_assert_eq!(up_off.len(), slots);
    debug_assert_eq!(down_w.len(), slots);
    debug_assert_eq!(down_off.len(), slots);
    let profile_start = (std::env::var("RNB_METAL_GLM_MOE_PROFILE").as_deref() == Ok("1"))
        .then(std::time::Instant::now);

    carrier.upload_input(input);
    carrier.upload_offsets(gate_off, up_off, down_off);
    carrier.upload_route_weights(route_weights);
    let t_upload = profile_start.map(|s| s.elapsed().as_secs_f64() * 1000.0);
    let stage_profile = std::env::var("RNB_METAL_GLM_MOE_STAGE_PROFILE").as_deref() == Ok("1");
    let mut cmd = ctx.queue.commandBuffer().expect("command buffer");
    let mut enc = cmd.computeCommandEncoder().expect("compute encoder");

    let gate_up_pipeline = if select.gate_up_iq2s {
        ctx.glm_moe_decode_iq2s_selected_slots_pipeline()
    } else {
        ctx.glm_moe_decode_iq2xxs_selected_slots_pipeline()
    };
    encode_glm_moe_decode_iq_selected_slots(
        gate_up_pipeline,
        &enc,
        &gate_w[..sparse_slots],
        &gate_off[..sparse_slots],
        &carrier.input_dev,
        &carrier.gate_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_ff,
        sparse_slots,
    );
    encode_glm_moe_decode_iq_selected_slots(
        gate_up_pipeline,
        &enc,
        &up_w[..sparse_slots],
        &up_off[..sparse_slots],
        &carrier.input_dev,
        &carrier.up_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_ff,
        sparse_slots,
    );

    let shared_slot = sparse_slots;
    let f32_bytes = std::mem::size_of::<f32>();
    let shared_ff_offset = shared_slot * carrier.n_ff * f32_bytes;
    if stage_profile {
        glm_moe_stage_cut(ctx, &mut cmd, &mut enc, 0);
    }
    let encode_shared_gate_up = if select.shared_gate_up_q6k {
        crate::compute::encode_gemv_q6k_auto_offset
    } else {
        crate::compute::encode_gemv_q5k_auto_offset
    };
    encode_shared_gate_up(
        ctx,
        &enc,
        &gate_w[shared_slot],
        &carrier.input_dev,
        &carrier.gate_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.gate_off_buf[shared_slot],
        0,
        shared_ff_offset,
        carrier.n_ff,
    );
    encode_shared_gate_up(
        ctx,
        &enc,
        &up_w[shared_slot],
        &carrier.input_dev,
        &carrier.up_dev,
        &carrier.n_ff_buf,
        &carrier.n_embd_buf,
        &carrier.up_off_buf[shared_slot],
        0,
        shared_ff_offset,
        carrier.n_ff,
    );
    if stage_profile {
        glm_moe_stage_cut(ctx, &mut cmd, &mut enc, 1);
    }
    encode_silu_mul(
        ctx,
        &enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.total_ff_buf,
        carrier.slots * carrier.n_ff,
    );
    if stage_profile {
        glm_moe_stage_cut(ctx, &mut cmd, &mut enc, 2);
    }

    let down_pipeline = if select.down_iq4xs {
        ctx.glm_moe_decode_iq4xs_selected_slots_pipeline()
    } else {
        ctx.glm_moe_decode_iq3xxs_selected_slots_pipeline()
    };
    encode_glm_moe_decode_iq_selected_slots(
        down_pipeline,
        &enc,
        &down_w[..sparse_slots],
        &down_off[..sparse_slots],
        &carrier.gate_dev,
        &carrier.down_dev,
        &carrier.n_embd_buf,
        &carrier.n_ff_buf,
        &carrier.slots_buf,
        carrier.n_embd,
        sparse_slots,
    );
    if stage_profile {
        glm_moe_stage_cut(ctx, &mut cmd, &mut enc, 3);
    }
    let shared_embd_offset = shared_slot * carrier.n_embd * f32_bytes;
    if select.shared_down_q8_0 {
        let zero_off_buf = u32_buf(ctx, 0);
        crate::compute::encode_gemv_q8_0_at(
            ctx,
            &enc,
            &down_w[shared_slot],
            down_off[shared_slot] as usize,
            &carrier.gate_dev,
            shared_ff_offset,
            &carrier.down_dev,
            shared_embd_offset,
            &carrier.n_embd_buf,
            &carrier.n_ff_buf,
            &zero_off_buf,
            carrier.n_embd,
        );
    } else {
        crate::compute::encode_gemv_q6k_auto_offset(
            ctx,
            &enc,
            &down_w[shared_slot],
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.n_embd_buf,
            &carrier.n_ff_buf,
            &carrier.down_off_buf[shared_slot],
            shared_ff_offset,
            shared_embd_offset,
            carrier.n_embd,
        );
    }
    if stage_profile {
        glm_moe_stage_cut(ctx, &mut cmd, &mut enc, 4);
    }
    encode_qwen_moe_decode_reduce_slots(
        ctx,
        &enc,
        &carrier.down_dev,
        &carrier.route_weights_dev,
        &carrier.out_dev,
        &carrier.n_embd_buf,
        &carrier.slots_buf,
        carrier.n_embd,
    );
    if stage_profile {
        glm_moe_stage_cut(ctx, &mut cmd, &mut enc, 5);
    }

    enc.endEncoding();
    let t_encode = profile_start.map(|s| s.elapsed().as_secs_f64() * 1000.0);
    cmd.commit();
    let t_commit = profile_start.map(|s| s.elapsed().as_secs_f64() * 1000.0);
    cmd.waitUntilCompleted();
    if let Some(start) = profile_start {
        let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
        // kernel*Time = 드라이버(CPU) 처리 구간, GPUStart - kernelEnd = 스케줄링 갭.
        let drv_ms = (cmd.kernelEndTime() - cmd.kernelStartTime()) * 1000.0;
        let sched_ms = (cmd.GPUStartTime() - cmd.kernelEndTime()) * 1000.0;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let upload_ms = t_upload.unwrap_or(0.0);
        let encode_ms = t_encode.unwrap_or(0.0) - upload_ms;
        let commit_ms = t_commit.unwrap_or(0.0) - t_encode.unwrap_or(0.0);
        let wait_ms = wall_ms - t_commit.unwrap_or(0.0);
        GLM_MOE_DECODE_PROFILE.with(|profile| {
            let mut acc = profile.get();
            acc[0] += 1.0;
            acc[1] += gpu_ms;
            acc[2] += wall_ms;
            acc[3] += upload_ms;
            acc[4] += encode_ms;
            acc[5] += commit_ms;
            acc[6] += wait_ms;
            acc[7] += drv_ms;
            acc[8] += sched_ms;
            profile.set(acc);
            let n = acc[0];
            if (n as usize) % 75 == 0 {
                eprintln!(
                    "[metal-glm-moe-decode] calls={} wall={:.3} = upload={:.3} + encode={:.3} + commit={:.3} + wait={:.3} | gpu={:.3} drv={:.3} sched={:.3} (avg ms)",
                    n as usize,
                    acc[2] / n,
                    acc[3] / n,
                    acc[4] / n,
                    acc[5] / n,
                    acc[6] / n,
                    acc[1] / n,
                    acc[7] / n,
                    acc[8] / n,
                );
            }
        });
    }
    readback(&carrier.out_dev, carrier.n_embd)
}

/// pm112: GLM MLA decode chain scratch. front(q_a→rms→q_b + kv_a + k_b 64-head)
/// 와 back(v_b 64-head→o) 을 각각 단일 command buffer 로 묶는다 (wait 2회/layer).
/// shape 별 1회 alloc 후 재사용. `!Send+!Sync` 라 thread_local backend 소유.
pub(crate) struct GlmMlaCarrier {
    pub hidden: usize,
    pub q_rank: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub heads: usize,
    pub kv_rank: usize,
    pub q_nope: usize,
    pub value_dim: usize,
    input_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    qa_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    qa_norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    qabs_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    latent_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    concat_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_rank_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_nope_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_rank_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    value_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    zero_off_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl GlmMlaCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        hidden: usize,
        q_rank: usize,
        q_dim: usize,
        kv_dim: usize,
        heads: usize,
        kv_rank: usize,
        q_nope: usize,
        value_dim: usize,
        eps: f32,
    ) -> Self {
        Self {
            hidden,
            q_rank,
            q_dim,
            kv_dim,
            heads,
            kv_rank,
            q_nope,
            value_dim,
            input_dev: empty_f32_buf(ctx, hidden),
            qa_dev: empty_f32_buf(ctx, q_rank),
            qa_norm_dev: empty_f32_buf(ctx, q_rank),
            q_dev: empty_f32_buf(ctx, q_dim),
            kv_dev: empty_f32_buf(ctx, kv_dim),
            qabs_dev: empty_f32_buf(ctx, heads * kv_rank),
            latent_dev: empty_f32_buf(ctx, heads * kv_rank),
            concat_dev: empty_f32_buf(ctx, heads * value_dim),
            out_dev: empty_f32_buf(ctx, hidden),
            hidden_buf: u32_buf(ctx, hidden as u32),
            q_rank_buf: u32_buf(ctx, q_rank as u32),
            q_dim_buf: u32_buf(ctx, q_dim as u32),
            kv_dim_buf: u32_buf(ctx, kv_dim as u32),
            q_nope_buf: u32_buf(ctx, q_nope as u32),
            kv_rank_buf: u32_buf(ctx, kv_rank as u32),
            value_dim_buf: u32_buf(ctx, value_dim as u32),
            zero_off_buf: u32_buf(ctx, 0),
            eps_buf: f32_buf(ctx, eps),
        }
    }

    fn upload(buf: &ProtocolObject<dyn MTLBuffer>, values: &[f32]) {
        let contents = buf.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                contents.as_ptr() as *mut f32,
                values.len(),
            );
        }
    }

    fn read_into(buf: &ProtocolObject<dyn MTLBuffer>, out: &mut [f32]) {
        let contents = buf.contents();
        let slice: &[f32] =
            unsafe { std::slice::from_raw_parts(contents.as_ptr() as *const f32, out.len()) };
        out.copy_from_slice(slice);
    }
}

type WeightRef<'a> = (&'a ProtocolObject<dyn MTLBuffer>, u32);

/// pm112: MLA front chain — q_a(Q5K) → rms_norm → q_b(Q8_0), kv_a(Q8_0),
/// k_b(Q8_0 64-head, q_dev 를 head stride 로 직접 읽음) 를 단일 command buffer 로.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_mla_front_dispatch(
    ctx: &MetalContext,
    carrier: &GlmMlaCarrier,
    qa_w: WeightRef<'_>,
    qa_norm_w: WeightRef<'_>,
    qb_w: WeightRef<'_>,
    kva_w: WeightRef<'_>,
    kb_w: WeightRef<'_>,
    input: &[f32],
    q_out: &mut [f32],
    kv_out: &mut [f32],
    qabs_out: &mut [f32],
) {
    debug_assert_eq!(input.len(), carrier.hidden);
    debug_assert_eq!(q_out.len(), carrier.q_dim);
    debug_assert_eq!(kv_out.len(), carrier.kv_dim);
    debug_assert_eq!(qabs_out.len(), carrier.heads * carrier.kv_rank);
    GlmMlaCarrier::upload(&carrier.input_dev, input);
    let qa_off_buf = u32_buf(ctx, qa_w.1);
    let qb_off_buf = u32_buf(ctx, qb_w.1);
    let kva_off_buf = u32_buf(ctx, kva_w.1);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    crate::compute::encode_gemv_q5k_auto(
        ctx,
        &enc,
        qa_w.0,
        &carrier.input_dev,
        &carrier.qa_dev,
        &carrier.q_rank_buf,
        &carrier.hidden_buf,
        &qa_off_buf,
        carrier.q_rank,
    );
    encode_rms_norm_at(
        ctx,
        &enc,
        &carrier.qa_dev,
        qa_norm_w.0,
        qa_norm_w.1 as usize,
        &carrier.qa_norm_dev,
        &carrier.q_rank_buf,
        &carrier.eps_buf,
    );
    crate::compute::encode_gemv_q8_0(
        ctx,
        &enc,
        qb_w.0,
        &carrier.qa_norm_dev,
        &carrier.q_dev,
        &carrier.q_dim_buf,
        &carrier.q_rank_buf,
        &qb_off_buf,
        carrier.q_dim,
    );
    crate::compute::encode_gemv_q8_0(
        ctx,
        &enc,
        kva_w.0,
        &carrier.input_dev,
        &carrier.kv_dev,
        &carrier.kv_dim_buf,
        &carrier.hidden_buf,
        &kva_off_buf,
        carrier.kv_dim,
    );
    let qk_dim = carrier.q_dim / carrier.heads;
    let kb_head_bytes = carrier.kv_rank * (carrier.q_nope / 32) * 34;
    let f32_bytes = std::mem::size_of::<f32>();
    for head in 0..carrier.heads {
        crate::compute::encode_gemv_q8_0_at(
            ctx,
            &enc,
            kb_w.0,
            kb_w.1 as usize + head * kb_head_bytes,
            &carrier.q_dev,
            head * qk_dim * f32_bytes,
            &carrier.qabs_dev,
            head * carrier.kv_rank * f32_bytes,
            &carrier.kv_rank_buf,
            &carrier.q_nope_buf,
            &carrier.zero_off_buf,
            carrier.kv_rank,
        );
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    GlmMlaCarrier::read_into(&carrier.q_dev, q_out);
    GlmMlaCarrier::read_into(&carrier.kv_dev, kv_out);
    GlmMlaCarrier::read_into(&carrier.qabs_dev, qabs_out);
}

/// pm112: MLA back chain — v_b(Q8_0 64-head) → o(Q5K) 를 단일 command buffer 로.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_mla_back_dispatch(
    ctx: &MetalContext,
    carrier: &GlmMlaCarrier,
    vb_w: WeightRef<'_>,
    o_w: WeightRef<'_>,
    latent: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(latent.len(), carrier.heads * carrier.kv_rank);
    debug_assert_eq!(out.len(), carrier.hidden);
    GlmMlaCarrier::upload(&carrier.latent_dev, latent);
    let o_off_buf = u32_buf(ctx, o_w.1);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    let vb_head_bytes = carrier.value_dim * (carrier.kv_rank / 32) * 34;
    let f32_bytes = std::mem::size_of::<f32>();
    for head in 0..carrier.heads {
        crate::compute::encode_gemv_q8_0_at(
            ctx,
            &enc,
            vb_w.0,
            vb_w.1 as usize + head * vb_head_bytes,
            &carrier.latent_dev,
            head * carrier.kv_rank * f32_bytes,
            &carrier.concat_dev,
            head * carrier.value_dim * f32_bytes,
            &carrier.value_dim_buf,
            &carrier.kv_rank_buf,
            &carrier.zero_off_buf,
            carrier.value_dim,
        );
    }
    // q_dim == heads * value_dim == concat 길이 (o 의 K)
    crate::compute::encode_gemv_q5k_auto(
        ctx,
        &enc,
        o_w.0,
        &carrier.concat_dev,
        &carrier.out_dev,
        &carrier.hidden_buf,
        &carrier.q_dim_buf,
        &o_off_buf,
        carrier.hidden,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    GlmMlaCarrier::read_into(&carrier.out_dev, out);
}

/// pm112: rms_norm encode 의 weight 바인딩 오프셋 변형 (resident wrap 용).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rms_norm_at(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_buf: &ProtocolObject<dyn MTLBuffer>,
    weight_byte_offset: usize,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    dim_buf: &ProtocolObject<dyn MTLBuffer>,
    eps_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    enc.setComputePipelineState(&ctx.rms_norm_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(weight_buf), weight_byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dim_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(eps_buf), 0, 4);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: RMS_TG_SIZE,
            height: 1,
            depth: 1,
        },
    );
}

impl PrefillFfnCarrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        hidden_dim: usize,
        ffn_dim: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            hidden_dim,
            ffn_dim,
            seq_len,
            normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            gate_dev: empty_f32_buf(ctx, seq_len * ffn_dim),
            up_dev: empty_f32_buf(ctx, seq_len * ffn_dim),
            down_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            hdim_buf: u32_buf(ctx, hidden_dim as u32),
            fdim_buf: u32_buf(ctx, ffn_dim as u32),
            k_hidden_buf: u32_buf(ctx, hidden_dim as u32),
            k_ffn_buf: u32_buf(ctx, ffn_dim as u32),
            m_buf: u32_buf(ctx, seq_len as u32),
            act_dim_buf: u32_buf(ctx, (seq_len * ffn_dim) as u32),
            normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            act_f16_dev: empty_f16_buf(ctx, seq_len * ffn_dim),
            normed_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
        }
    }

    pub(crate) fn new_private_intermediates(
        ctx: &MetalContext,
        hidden_dim: usize,
        ffn_dim: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            hidden_dim,
            ffn_dim,
            seq_len,
            normed_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            gate_dev: private_f32_buf(ctx, seq_len * ffn_dim),
            up_dev: private_f32_buf(ctx, seq_len * ffn_dim),
            down_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            hdim_buf: u32_buf(ctx, hidden_dim as u32),
            fdim_buf: u32_buf(ctx, ffn_dim as u32),
            k_hidden_buf: u32_buf(ctx, hidden_dim as u32),
            k_ffn_buf: u32_buf(ctx, ffn_dim as u32),
            m_buf: u32_buf(ctx, seq_len as u32),
            act_dim_buf: u32_buf(ctx, (seq_len * ffn_dim) as u32),
            normed_f16_dev: private_f16_buf(ctx, seq_len * hidden_dim),
            act_f16_dev: private_f16_buf(ctx, seq_len * ffn_dim),
            normed_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
        }
    }

    /// normed host slice[seq_len*hidden_dim] → normed_dev 업로드.
    fn upload_normed(&self, normed: &[f32]) {
        debug_assert_eq!(normed.len(), self.seq_len * self.hidden_dim);
        let contents = self.normed_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                normed.as_ptr(),
                contents.as_ptr() as *mut f32,
                self.seq_len * self.hidden_dim,
            );
        }
    }
}

pub(crate) fn prefill_ffn_chain_v2_scatter_supported(
    ctx: &MetalContext,
    down_is_q6k: bool,
) -> bool {
    ctx.tensorops_capable
        && ctx.gemm_q4k_tensorops_v2_pipeline.is_some()
        && ctx.cast_f32_f16_pipeline.is_some()
        && (!down_is_q6k || ctx.gemm_q6k_tensorops_v2_pipeline.is_some())
        && std::env::var("RNB_METAL_PREFILL_FFN_V2").as_deref() != Ok("0")
        && std::env::var("RNB_METAL_PREFILL_FFN_KERNEL").as_deref() != Ok("naive")
}

pub(crate) fn qwen_moe_prefill_id_matmul_supported(ctx: &MetalContext, down_is_q6k: bool) -> bool {
    down_is_q6k
        && ctx.tensorops_capable
        && ctx.gemm_q4k_tensorops_id_pipeline.is_some()
        && ctx.gemm_q6k_tensorops_id_pipeline.is_some()
}

pub(crate) fn qwen_moe_prefill_id_gate_up_supported(ctx: &MetalContext, down_is_q6k: bool) -> bool {
    ctx.tensorops_capable
        && ctx.gemm_q4k_tensorops_id_pipeline.is_some()
        && ctx.gemm_q4k_tensorops_v2_pipeline.is_some()
        && (!down_is_q6k || ctx.gemm_q6k_tensorops_v2_pipeline.is_some())
}

pub(crate) fn qwen_moe_prefill_id_gate_up_f16_supported(
    ctx: &MetalContext,
    down_is_q6k: bool,
) -> bool {
    ctx.tensorops_capable
        && ctx.gemm_q4k_tensorops_id_v2_f16_pipeline.is_some()
        && ctx.gemm_q4k_tensorops_v2_pipeline.is_some()
        && (!down_is_q6k || ctx.gemm_q6k_tensorops_v2_pipeline.is_some())
}

pub(crate) fn qwen_moe_prefill_mulmmid_v3_supported(ctx: &MetalContext, down_is_q6k: bool) -> bool {
    ctx.tensorops_capable
        && ctx.gemm_q4k_tensorops_id_v2_f16_pipeline.is_some()
        && if down_is_q6k {
            ctx.qwen_moe_v3_q6_down_pipeline.is_some()
        } else {
            ctx.qwen_moe_v3_q4_down_pipeline.is_some()
        }
        && ctx.qwen_moe_v3_token_rank_combine_pipeline.is_some()
}

pub(crate) fn qwen_moe_prefill_mulmmid_v4_supported(ctx: &MetalContext, down_is_q6k: bool) -> bool {
    ctx.tensorops_capable
        && ctx.gemm_q4k_tensorops_id_v2_f16_pipeline.is_some()
        && if down_is_q6k {
            ctx.qwen_moe_v4_q6_down_scatter_pipeline.is_some()
        } else {
            ctx.qwen_moe_v4_q4_down_scatter_pipeline.is_some()
        }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_id_gate_up_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoePrefillIdGateUpCarrier,
    norm_all_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_off: u32,
    up_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_all_off: u32,
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    block_count: usize,
    hidden_dim: usize,
) {
    debug_assert!(ctx.tensorops_capable);
    debug_assert!(ctx.gemm_q4k_tensorops_id_pipeline.is_some());
    let ffn_dim = carrier.ffn_dim;
    crate::compute::encode_gemm_q4k_tensorops_id(
        ctx,
        enc,
        gate_all_w_buf,
        gate_all_off,
        norm_all_buf,
        &carrier.gate_dev,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        ffn_dim,
        hidden_dim,
        block_count,
        gate_expert_bytes,
    );
    crate::compute::encode_gemm_q4k_tensorops_id(
        ctx,
        enc,
        up_all_w_buf,
        up_all_off,
        norm_all_buf,
        &carrier.up_dev,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        ffn_dim,
        hidden_dim,
        block_count,
        up_expert_bytes,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_id_gate_up_f16_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoePrefillIdGateUpF16Carrier,
    norm_all_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_off: u32,
    up_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_all_off: u32,
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    block_count: usize,
    hidden_dim: usize,
) {
    debug_assert!(ctx.tensorops_capable);
    debug_assert!(ctx.gemm_q4k_tensorops_id_v2_f16_pipeline.is_some());
    let ffn_dim = carrier.ffn_dim;
    crate::compute::encode_gemm_q4k_tensorops_id_f16(
        ctx,
        enc,
        gate_all_w_buf,
        gate_all_off,
        norm_all_buf,
        &carrier.gate_dev,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        ffn_dim,
        hidden_dim,
        block_count,
        gate_expert_bytes,
    );
    crate::compute::encode_gemm_q4k_tensorops_id_f16(
        ctx,
        enc,
        up_all_w_buf,
        up_all_off,
        norm_all_buf,
        &carrier.up_dev,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        ffn_dim,
        hidden_dim,
        block_count,
        up_expert_bytes,
    );
}

fn encode_silu_mul_to_f16_slice(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    id_carrier: &QwenMoePrefillIdGateUpCarrier,
    group_carrier: &PrefillFfnCarrier,
    group_start: usize,
    group_len: usize,
) {
    let ffn_dim = id_carrier.ffn_dim;
    let elem_offset = group_start
        .checked_mul(ffn_dim)
        .expect("Qwen MoE id gate/up slice offset overflow");
    let byte_offset = elem_offset
        .checked_mul(std::mem::size_of::<f32>())
        .expect("Qwen MoE id gate/up byte offset overflow");
    let total = group_len
        .checked_mul(ffn_dim)
        .expect("Qwen MoE id gate/up slice element count overflow");
    enc.setComputePipelineState(&ctx.silu_mul_f16_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&id_carrier.gate_dev), byte_offset, 0);
        enc.setBuffer_offset_atIndex(Some(&id_carrier.up_dev), byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(&group_carrier.act_f16_dev), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&group_carrier.act_dim_buf), 0, 3);
    }
    let tgw = ctx.silu_mul_f16_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

fn encode_silu_mul_half_to_f16_slice(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    id_carrier: &QwenMoePrefillIdGateUpF16Carrier,
    group_carrier: &PrefillFfnCarrier,
    group_start: usize,
    group_len: usize,
) {
    let ffn_dim = id_carrier.ffn_dim;
    let elem_offset = group_start
        .checked_mul(ffn_dim)
        .expect("Qwen MoE id gate/up f16 slice offset overflow");
    let byte_offset = elem_offset
        .checked_mul(std::mem::size_of::<u16>())
        .expect("Qwen MoE id gate/up f16 byte offset overflow");
    let total = group_len
        .checked_mul(ffn_dim)
        .expect("Qwen MoE id gate/up f16 slice element count overflow");
    enc.setComputePipelineState(&ctx.silu_mul_half_f16_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&id_carrier.gate_dev), byte_offset, 0);
        enc.setBuffer_offset_atIndex(Some(&id_carrier.up_dev), byte_offset, 1);
        enc.setBuffer_offset_atIndex(Some(&group_carrier.act_f16_dev), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&group_carrier.act_dim_buf), 0, 3);
    }
    let tgw = ctx.silu_mul_half_f16_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

fn encode_silu_mul_half_inplace(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    up_buf: &ProtocolObject<dyn MTLBuffer>,
    elems: usize,
) -> Result<(), QwenMoeIdPreflightReason> {
    let total =
        u32::try_from(elems).map_err(|_| QwenMoeIdPreflightReason::InvalidPrimitiveShape)?;
    enc.setComputePipelineState(&ctx.silu_mul_half_f16_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 2);
    }
    set_u32_bytes(enc, total, 3);
    let tgw = ctx.silu_mul_half_f16_pipeline.threadExecutionWidth().max(1);
    let grid = MTLSize {
        width: elems.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_mulmmid_v3_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeMulMmIdV3Carrier,
    accum: &QwenMoePrefillAccumCarrier,
    norm_all_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_off: u32,
    up_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_all_off: u32,
    down_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_all_off: u32,
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    token_ids_sorted_buf: &ProtocolObject<dyn MTLBuffer>,
    dst_slots_sorted_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_token_rank_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    block_count: usize,
    down_is_q6k: bool,
) -> Result<(), QwenMoeIdPreflightReason> {
    if !qwen_moe_prefill_mulmmid_v3_supported(ctx, down_is_q6k) {
        return Err(QwenMoeIdPreflightReason::PipelineUnavailable);
    }
    crate::compute::encode_gemm_q4k_tensorops_id_f16(
        ctx,
        enc,
        gate_all_w_buf,
        gate_all_off,
        norm_all_buf,
        &carrier.gate_half_dev,
        token_ids_sorted_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        carrier.ffn_dim,
        carrier.hidden_dim,
        block_count,
        gate_expert_bytes,
    );
    crate::compute::encode_gemm_q4k_tensorops_id_f16(
        ctx,
        enc,
        up_all_w_buf,
        up_all_off,
        norm_all_buf,
        &carrier.up_half_dev,
        token_ids_sorted_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        carrier.ffn_dim,
        carrier.hidden_dim,
        block_count,
        up_expert_bytes,
    );
    encode_silu_mul_half_inplace(
        ctx,
        enc,
        &carrier.gate_half_dev,
        &carrier.up_half_dev,
        carrier.slots * carrier.ffn_dim,
    )?;
    if down_is_q6k {
        crate::compute::encode_qwen_moe_v3_q6_down(
            ctx,
            enc,
            down_all_w_buf,
            down_all_off,
            &carrier.gate_half_dev,
            &carrier.down_token_rank_dev,
            dst_slots_sorted_buf,
            expert_offsets_buf,
            expert_counts_buf,
            block_experts_buf,
            block_local0_buf,
            carrier.hidden_dim,
            carrier.ffn_dim,
            block_count,
            down_expert_bytes,
        )?;
    } else {
        crate::compute::encode_qwen_moe_v3_q4_down(
            ctx,
            enc,
            down_all_w_buf,
            down_all_off,
            &carrier.gate_half_dev,
            &carrier.down_token_rank_dev,
            dst_slots_sorted_buf,
            expert_offsets_buf,
            expert_counts_buf,
            block_experts_buf,
            block_local0_buf,
            carrier.hidden_dim,
            carrier.ffn_dim,
            block_count,
            down_expert_bytes,
        )?;
    }
    crate::compute::encode_qwen_moe_v3_token_rank_combine(
        ctx,
        enc,
        &carrier.down_token_rank_dev,
        route_weights_token_rank_buf,
        &accum.out_dev,
        carrier.seq_len,
        carrier.n_expert_used,
        carrier.hidden_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_mulmmid_v4_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoeMulMmIdV4Carrier,
    accum: &QwenMoePrefillAccumCarrier,
    norm_all_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_off: u32,
    up_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_all_off: u32,
    down_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_all_off: u32,
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    token_ids_sorted_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_sorted_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    id_block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    id_block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    id_block_count: usize,
    expert_rank_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_rank_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    rank_block_offsets: &[u32],
    rank_block_counts: &[u32],
    down_is_q6k: bool,
) -> Result<(), QwenMoeIdPreflightReason> {
    if !qwen_moe_prefill_mulmmid_v4_supported(ctx, down_is_q6k) {
        return Err(QwenMoeIdPreflightReason::PipelineUnavailable);
    }
    if rank_block_offsets.len() != carrier.n_expert_used
        || rank_block_counts.len() != carrier.n_expert_used
    {
        return Err(QwenMoeIdPreflightReason::RankCountMismatch);
    }
    debug_assert_eq!(carrier.hidden_dim, accum.hidden_dim);
    debug_assert_eq!(carrier.slots, accum.seq_len * carrier.n_expert_used);
    crate::compute::encode_gemm_q4k_tensorops_id_f16(
        ctx,
        enc,
        gate_all_w_buf,
        gate_all_off,
        norm_all_buf,
        &carrier.gate_half_dev,
        token_ids_sorted_buf,
        expert_offsets_buf,
        expert_counts_buf,
        id_block_experts_buf,
        id_block_local0_buf,
        carrier.ffn_dim,
        carrier.hidden_dim,
        id_block_count,
        gate_expert_bytes,
    );
    crate::compute::encode_gemm_q4k_tensorops_id_f16(
        ctx,
        enc,
        up_all_w_buf,
        up_all_off,
        norm_all_buf,
        &carrier.up_half_dev,
        token_ids_sorted_buf,
        expert_offsets_buf,
        expert_counts_buf,
        id_block_experts_buf,
        id_block_local0_buf,
        carrier.ffn_dim,
        carrier.hidden_dim,
        id_block_count,
        up_expert_bytes,
    );
    encode_silu_mul_half_inplace(
        ctx,
        enc,
        &carrier.gate_half_dev,
        &carrier.up_half_dev,
        carrier.slots * carrier.ffn_dim,
    )?;
    for rank in 0..carrier.n_expert_used {
        let rank_block_offset = rank_block_offsets[rank] as usize;
        let rank_block_count = rank_block_counts[rank] as usize;
        if down_is_q6k {
            crate::compute::encode_qwen_moe_v4_q6_down_scatter(
                ctx,
                enc,
                down_all_w_buf,
                down_all_off,
                &carrier.gate_half_dev,
                &accum.out_dev,
                token_ids_sorted_buf,
                route_weights_sorted_buf,
                expert_rank_offsets_buf,
                expert_rank_counts_buf,
                rank_block_experts_buf,
                rank_block_local0_buf,
                rank_block_offset,
                rank_block_count,
                rank,
                carrier.n_expert_used,
                carrier.hidden_dim,
                carrier.ffn_dim,
                down_expert_bytes,
            )?;
        } else {
            crate::compute::encode_qwen_moe_v4_q4_down_scatter(
                ctx,
                enc,
                down_all_w_buf,
                down_all_off,
                &carrier.gate_half_dev,
                &accum.out_dev,
                token_ids_sorted_buf,
                route_weights_sorted_buf,
                expert_rank_offsets_buf,
                expert_rank_counts_buf,
                rank_block_experts_buf,
                rank_block_local0_buf,
                rank_block_offset,
                rank_block_count,
                rank,
                carrier.n_expert_used,
                carrier.hidden_dim,
                carrier.ffn_dim,
                down_expert_bytes,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_id_gate_up_down_scatter_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    id_carrier: &QwenMoePrefillIdGateUpCarrier,
    group_carrier: &PrefillFfnCarrier,
    accum: &QwenMoePrefillAccumCarrier,
    down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_off: u32,
    down_is_q6k: bool,
    group_start: u32,
    group_len: usize,
    total_elems: u32,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    debug_assert!(prefill_ffn_chain_v2_scatter_supported(ctx, down_is_q6k));
    debug_assert_eq!(group_carrier.seq_len, group_len);
    let hidden_dim = group_carrier.hidden_dim;

    encode_silu_mul_to_f16_slice(
        ctx,
        enc,
        id_carrier,
        group_carrier,
        group_start as usize,
        group_len,
    );
    let down_scatter_supported = qwen_moe_prefill_down_scatter_requested()
        && ctx.gemm_q4k_tensorops_v2_scatter_accum_pipeline.is_some()
        && (!down_is_q6k || ctx.gemm_q6k_tensorops_v2_scatter_accum_pipeline.is_some());
    if down_scatter_supported {
        if down_is_q6k {
            crate::compute::encode_gemm_q6k_tensorops_v2_scatter_accum(
                ctx,
                enc,
                down_w_buf,
                down_off,
                &group_carrier.act_f16_dev,
                &accum.out_dev,
                token_ids_buf,
                route_weights_buf,
                hidden_dim,
                id_carrier.ffn_dim,
                group_len,
                group_start,
            );
        } else {
            crate::compute::encode_gemm_q4k_tensorops_v2_scatter_accum(
                ctx,
                enc,
                down_w_buf,
                down_off,
                &group_carrier.act_f16_dev,
                &accum.out_dev,
                token_ids_buf,
                route_weights_buf,
                hidden_dim,
                id_carrier.ffn_dim,
                group_len,
                group_start,
            );
        }
        return;
    }
    if down_is_q6k {
        crate::compute::encode_gemm_q6k_tensorops_v2(
            ctx,
            enc,
            down_w_buf,
            down_off,
            &group_carrier.act_f16_dev,
            &group_carrier.down_dev,
            &group_carrier.hdim_buf,
            &group_carrier.k_ffn_buf,
            &group_carrier.m_buf,
            hidden_dim,
            group_len,
        );
    } else {
        crate::compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            enc,
            down_w_buf,
            down_off,
            &group_carrier.act_f16_dev,
            &group_carrier.down_dev,
            &group_carrier.hdim_buf,
            &group_carrier.k_ffn_buf,
            &group_carrier.m_buf,
            hidden_dim,
            group_len,
        );
    }

    enc.setComputePipelineState(&ctx.qwen_moe_prefill_scatter_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&group_carrier.down_dev), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&accum.out_dev), 0, 1);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(route_weights_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&accum.hidden_buf), 0, 4);
    }
    set_u32_bytes(enc, group_start, 5);
    set_u32_bytes(enc, total_elems, 6);
    let tgw = ctx
        .qwen_moe_prefill_scatter_pipeline
        .threadExecutionWidth()
        .max(1);
    let total = group_len * hidden_dim;
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_id_gate_up_f16_down_scatter_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    id_carrier: &QwenMoePrefillIdGateUpF16Carrier,
    group_carrier: &PrefillFfnCarrier,
    accum: &QwenMoePrefillAccumCarrier,
    down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_off: u32,
    down_is_q6k: bool,
    group_start: u32,
    group_len: usize,
    total_elems: u32,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    debug_assert!(prefill_ffn_chain_v2_scatter_supported(ctx, down_is_q6k));
    debug_assert_eq!(group_carrier.seq_len, group_len);
    let hidden_dim = group_carrier.hidden_dim;

    encode_silu_mul_half_to_f16_slice(
        ctx,
        enc,
        id_carrier,
        group_carrier,
        group_start as usize,
        group_len,
    );
    let down_scatter_supported = qwen_moe_prefill_down_scatter_requested()
        && ctx.gemm_q4k_tensorops_v2_scatter_accum_pipeline.is_some()
        && (!down_is_q6k || ctx.gemm_q6k_tensorops_v2_scatter_accum_pipeline.is_some());
    if down_scatter_supported {
        if down_is_q6k {
            crate::compute::encode_gemm_q6k_tensorops_v2_scatter_accum(
                ctx,
                enc,
                down_w_buf,
                down_off,
                &group_carrier.act_f16_dev,
                &accum.out_dev,
                token_ids_buf,
                route_weights_buf,
                hidden_dim,
                id_carrier.ffn_dim,
                group_len,
                group_start,
            );
        } else {
            crate::compute::encode_gemm_q4k_tensorops_v2_scatter_accum(
                ctx,
                enc,
                down_w_buf,
                down_off,
                &group_carrier.act_f16_dev,
                &accum.out_dev,
                token_ids_buf,
                route_weights_buf,
                hidden_dim,
                id_carrier.ffn_dim,
                group_len,
                group_start,
            );
        }
        return;
    }
    if down_is_q6k {
        crate::compute::encode_gemm_q6k_tensorops_v2(
            ctx,
            enc,
            down_w_buf,
            down_off,
            &group_carrier.act_f16_dev,
            &group_carrier.down_dev,
            &group_carrier.hdim_buf,
            &group_carrier.k_ffn_buf,
            &group_carrier.m_buf,
            hidden_dim,
            group_len,
        );
    } else {
        crate::compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            enc,
            down_w_buf,
            down_off,
            &group_carrier.act_f16_dev,
            &group_carrier.down_dev,
            &group_carrier.hdim_buf,
            &group_carrier.k_ffn_buf,
            &group_carrier.m_buf,
            hidden_dim,
            group_len,
        );
    }

    enc.setComputePipelineState(&ctx.qwen_moe_prefill_scatter_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&group_carrier.down_dev), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&accum.out_dev), 0, 1);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(route_weights_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&accum.hidden_buf), 0, 4);
    }
    set_u32_bytes(enc, group_start, 5);
    set_u32_bytes(enc, total_elems, 6);
    let tgw = ctx
        .qwen_moe_prefill_scatter_pipeline
        .threadExecutionWidth()
        .max(1);
    let total = group_len * hidden_dim;
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_moe_prefill_id_matmul_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &QwenMoePrefillIdCarrier,
    accum: &QwenMoePrefillAccumCarrier,
    norm_all_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_all_off: u32,
    up_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_all_off: u32,
    down_all_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_all_off: u32,
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_counts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_experts_buf: &ProtocolObject<dyn MTLBuffer>,
    block_local0_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_offsets: &[u32],
    expert_counts: &[u32],
    block_count: usize,
    n_expert: usize,
) {
    debug_assert!(qwen_moe_prefill_id_matmul_supported(ctx, true));
    debug_assert_eq!(expert_offsets.len(), n_expert);
    debug_assert_eq!(expert_counts.len(), n_expert);
    let hidden_dim = carrier.hidden_dim;
    let ffn_dim = carrier.ffn_dim;

    crate::compute::encode_gemm_q4k_tensorops_id(
        ctx,
        enc,
        gate_all_w_buf,
        gate_all_off,
        norm_all_buf,
        &carrier.gate_dev,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        ffn_dim,
        hidden_dim,
        block_count,
        gate_expert_bytes,
    );
    crate::compute::encode_gemm_q4k_tensorops_id(
        ctx,
        enc,
        up_all_w_buf,
        up_all_off,
        norm_all_buf,
        &carrier.up_dev,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        ffn_dim,
        hidden_dim,
        block_count,
        up_expert_bytes,
    );
    encode_silu_mul(
        ctx,
        enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.total_ff_buf,
        carrier.slots * ffn_dim,
    );
    crate::compute::encode_gemm_q6k_tensorops_id(
        ctx,
        enc,
        down_all_w_buf,
        down_all_off,
        &carrier.gate_dev,
        &carrier.down_dev,
        token_ids_buf,
        expert_offsets_buf,
        expert_counts_buf,
        block_experts_buf,
        block_local0_buf,
        hidden_dim,
        ffn_dim,
        block_count,
        down_expert_bytes,
    );

    for expert in 0..n_expert {
        let count = expert_counts[expert] as usize;
        if count == 0 {
            continue;
        }
        let group_start = expert_offsets[expert] as usize;
        let total_elems = count
            .checked_mul(hidden_dim)
            .and_then(|v| u32::try_from(v).ok())
            .expect("Qwen MoE id scatter element count overflow");
        let down_byte_offset = group_start
            .checked_mul(hidden_dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .expect("Qwen MoE id scatter down offset overflow");
        enc.setComputePipelineState(&ctx.qwen_moe_prefill_scatter_pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&carrier.down_dev), down_byte_offset, 0);
            enc.setBuffer_offset_atIndex(Some(&accum.out_dev), 0, 1);
            enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(route_weights_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&accum.hidden_buf), 0, 4);
        }
        set_u32_bytes(enc, group_start as u32, 5);
        set_u32_bytes(enc, total_elems, 6);
        let tgw = ctx
            .qwen_moe_prefill_scatter_pipeline
            .threadExecutionWidth()
            .max(1);
        let grid = MTLSize {
            width: (count * hidden_dim).div_ceil(tgw),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tgw,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
}

#[allow(clippy::too_many_arguments)]
fn prefill_ffn_chain_v2_scatter_accum_encode_from_input(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillFfnCarrier,
    accum: &QwenMoePrefillAccumCarrier,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off: u32,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_off: u32,
    down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_off: u32,
    down_is_q6k: bool,
    m: usize,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    group_start: u32,
    total_elems: u32,
) {
    debug_assert!(prefill_ffn_chain_v2_scatter_supported(ctx, down_is_q6k));
    let hidden_dim = carrier.hidden_dim;
    debug_assert_eq!(m, carrier.seq_len);

    crate::compute::encode_cast_f32_to_f16(
        ctx,
        enc,
        &carrier.normed_dev,
        &carrier.normed_f16_dev,
        &carrier.normed_elems_buf,
        m * hidden_dim,
    );

    prefill_ffn_chain_v2_scatter_accum_encode_from_f16(
        ctx,
        enc,
        carrier,
        accum,
        &carrier.normed_f16_dev,
        gate_w_buf,
        gate_off,
        up_w_buf,
        up_off,
        down_w_buf,
        down_off,
        down_is_q6k,
        m,
        token_ids_buf,
        route_weights_buf,
        group_start,
        total_elems,
    );
}

#[allow(clippy::too_many_arguments)]
fn prefill_ffn_chain_v2_scatter_accum_encode_from_f16(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillFfnCarrier,
    accum: &QwenMoePrefillAccumCarrier,
    input_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off: u32,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_off: u32,
    down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_off: u32,
    down_is_q6k: bool,
    m: usize,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    group_start: u32,
    total_elems: u32,
) {
    debug_assert!(prefill_ffn_chain_v2_scatter_supported(ctx, down_is_q6k));
    let hidden_dim = carrier.hidden_dim;
    let ffn_dim = carrier.ffn_dim;
    debug_assert_eq!(m, carrier.seq_len);

    let used_gate_up_pair = crate::compute::encode_gemm_q4k_tensorops_v2_pair(
        ctx,
        enc,
        gate_w_buf,
        gate_off,
        up_w_buf,
        up_off,
        input_f16_buf,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.fdim_buf,
        &carrier.k_hidden_buf,
        &carrier.m_buf,
        ffn_dim,
        m,
    );
    if !used_gate_up_pair {
        crate::compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            enc,
            gate_w_buf,
            gate_off,
            input_f16_buf,
            &carrier.gate_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
        crate::compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            enc,
            up_w_buf,
            up_off,
            input_f16_buf,
            &carrier.up_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
    }
    crate::compute::encode_silu_mul_to_f16(
        ctx,
        enc,
        &carrier.gate_dev,
        &carrier.up_dev,
        &carrier.act_f16_dev,
        &carrier.act_dim_buf,
        m * ffn_dim,
    );
    let down_scatter_supported = qwen_moe_prefill_down_scatter_requested()
        && ctx.gemm_q4k_tensorops_v2_scatter_accum_pipeline.is_some()
        && (!down_is_q6k || ctx.gemm_q6k_tensorops_v2_scatter_accum_pipeline.is_some());
    if down_scatter_supported {
        if down_is_q6k {
            crate::compute::encode_gemm_q6k_tensorops_v2_scatter_accum(
                ctx,
                enc,
                down_w_buf,
                down_off,
                &carrier.act_f16_dev,
                &accum.out_dev,
                token_ids_buf,
                route_weights_buf,
                hidden_dim,
                ffn_dim,
                m,
                group_start,
            );
        } else {
            crate::compute::encode_gemm_q4k_tensorops_v2_scatter_accum(
                ctx,
                enc,
                down_w_buf,
                down_off,
                &carrier.act_f16_dev,
                &accum.out_dev,
                token_ids_buf,
                route_weights_buf,
                hidden_dim,
                ffn_dim,
                m,
                group_start,
            );
        }
        return;
    }
    if down_is_q6k {
        crate::compute::encode_gemm_q6k_tensorops_v2(
            ctx,
            enc,
            down_w_buf,
            down_off,
            &carrier.act_f16_dev,
            &carrier.down_dev,
            &carrier.hdim_buf,
            &carrier.k_ffn_buf,
            &carrier.m_buf,
            hidden_dim,
            m,
        );
    } else {
        crate::compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            enc,
            down_w_buf,
            down_off,
            &carrier.act_f16_dev,
            &carrier.down_dev,
            &carrier.hdim_buf,
            &carrier.k_ffn_buf,
            &carrier.m_buf,
            hidden_dim,
            m,
        );
    }

    enc.setComputePipelineState(&ctx.qwen_moe_prefill_scatter_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&carrier.down_dev), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&accum.out_dev), 0, 1);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(route_weights_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&accum.hidden_buf), 0, 4);
    }
    set_u32_bytes(enc, group_start, 5);
    set_u32_bytes(enc, total_elems, 6);
    let tgw = ctx
        .qwen_moe_prefill_scatter_pipeline
        .threadExecutionWidth()
        .max(1);
    let total = m * hidden_dim;
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_ffn_chain_v2_scatter_accum_encode(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillFfnCarrier,
    accum: &QwenMoePrefillAccumCarrier,
    normed: &[f32],
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off: u32,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_off: u32,
    down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_off: u32,
    down_is_q6k: bool,
    m: usize,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    group_start: u32,
    total_elems: u32,
) {
    carrier.upload_normed(normed);
    prefill_ffn_chain_v2_scatter_accum_encode_from_input(
        ctx,
        enc,
        carrier,
        accum,
        gate_w_buf,
        gate_off,
        up_w_buf,
        up_off,
        down_w_buf,
        down_off,
        down_is_q6k,
        m,
        token_ids_buf,
        route_weights_buf,
        group_start,
        total_elems,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_ffn_chain_v2_scatter_accum_encode_gather(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &PrefillFfnCarrier,
    accum: &QwenMoePrefillAccumCarrier,
    norm_all_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off: u32,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_off: u32,
    down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_off: u32,
    down_is_q6k: bool,
    m: usize,
    token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
    route_weights_buf: &ProtocolObject<dyn MTLBuffer>,
    group_start: u32,
    total_elems: u32,
) {
    debug_assert_eq!(m, carrier.seq_len);

    if qwen_moe_prefill_gather_f16_requested() {
        enc.setComputePipelineState(&ctx.qwen_moe_prefill_gather_f16_pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(norm_all_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&carrier.normed_f16_dev), 0, 1);
            enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&carrier.hdim_buf), 0, 3);
        }
        set_u32_bytes(enc, group_start, 4);
        set_u32_bytes(enc, total_elems, 5);
        let tgw = ctx
            .qwen_moe_prefill_gather_f16_pipeline
            .threadExecutionWidth()
            .max(1);
        let total = m * carrier.hidden_dim;
        let grid = MTLSize {
            width: total.div_ceil(tgw),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tgw,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        prefill_ffn_chain_v2_scatter_accum_encode_from_f16(
            ctx,
            enc,
            carrier,
            accum,
            &carrier.normed_f16_dev,
            gate_w_buf,
            gate_off,
            up_w_buf,
            up_off,
            down_w_buf,
            down_off,
            down_is_q6k,
            m,
            token_ids_buf,
            route_weights_buf,
            group_start,
            total_elems,
        );
        return;
    }

    enc.setComputePipelineState(&ctx.qwen_moe_prefill_gather_pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(norm_all_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&carrier.normed_dev), 0, 1);
        enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&carrier.hdim_buf), 0, 3);
    }
    set_u32_bytes(enc, group_start, 4);
    set_u32_bytes(enc, total_elems, 5);
    let tgw = ctx
        .qwen_moe_prefill_gather_pipeline
        .threadExecutionWidth()
        .max(1);
    let total = m * carrier.hidden_dim;
    let grid = MTLSize {
        width: total.div_ceil(tgw),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tgw,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

    prefill_ffn_chain_v2_scatter_accum_encode_from_input(
        ctx,
        enc,
        carrier,
        accum,
        gate_w_buf,
        gate_off,
        up_w_buf,
        up_off,
        down_w_buf,
        down_off,
        down_is_q6k,
        m,
        token_ids_buf,
        route_weights_buf,
        group_start,
        total_elems,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_ffn_chain_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillFfnCarrier,
    normed: &[f32],
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
    up_w_buf: &ProtocolObject<dyn MTLBuffer>,
    up_off_buf: &ProtocolObject<dyn MTLBuffer>,
    down_w_buf: &ProtocolObject<dyn MTLBuffer>,
    down_off_buf: &ProtocolObject<dyn MTLBuffer>,
    down_is_q6k: bool,
    m: usize,
) -> Vec<f32> {
    let hidden_dim = carrier.hidden_dim;
    let ffn_dim = carrier.ffn_dim;
    carrier.upload_normed(normed);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");

    // pm34 M7: ABAB −29.6%(27B prefill 72.5→51.1s, 겹침0) + token-identical GREEN → default 승격.
    // capability(M5)면 GPU FFN 경로의 default 커널 = tensorops. RNB_METAL_PREFILL_FFN_KERNEL=naive
    // 로 opt-out(측정/디버그/회귀). 미-M5/구형 OS 는 capability false → 기존 naive/shared 보존.
    let force_naive = std::env::var("RNB_METAL_PREFILL_FFN_KERNEL").as_deref() == Ok("naive");
    // pm42 M3: v2 GEMM(llama 패턴 cooperative C, 64×128 winner 타일). activation 은 matmul2d f16
    // 강제라 normed/silu결과를 chain 1회 cast(carrier f16 버퍼). **default ON 승격**(pm42 측정:
    // 27B FFN GPU 16.0→1.79s = −88.8%/8.9x, prefill wall 30.4→16.3s = −46.4%, 짧은ko+긴en
    // token-identical). RNB_METAL_PREFILL_FFN_V2=0 으로 v1(64×32) opt-out. down Q6_K 면 q6k v2 필요.
    let use_tensorops_v2 = !force_naive
        && ctx.tensorops_capable
        && ctx.gemm_q4k_tensorops_v2_pipeline.is_some()
        && ctx.cast_f32_f16_pipeline.is_some()
        && (!down_is_q6k || ctx.gemm_q6k_tensorops_v2_pipeline.is_some())
        && std::env::var("RNB_METAL_PREFILL_FFN_V2").as_deref() != Ok("0");
    let use_tensorops = !use_tensorops_v2
        && !force_naive
        && ctx.tensorops_capable
        && ctx.gemm_q4k_tensorops_pipeline.is_some()
        && (!down_is_q6k || ctx.gemm_q6k_tensorops_pipeline.is_some());
    // pm33: default dequant-공유(shared) 커널. RNB_METAL_PREFILL_FFN_NAIVE=1 이면 naive(비교용).
    let naive = ctx.prefill_ffn_naive;
    // tensorops 경로: off_buf(u32 담은 MTLBuffer) 에서 page offset 을 host read → Metal-native
    // 인코더 offset 으로 적용(naive 는 off_buf 를 커널 buffer(5) 상수로 읽음 — 두 경로 동일 base).
    let read_off = |b: &ProtocolObject<dyn MTLBuffer>| -> u32 {
        unsafe { *(b.contents().as_ptr() as *const u32) }
    };
    // gate: normed[m,hid] → gate[m,ffn]  (N=ffn_dim, K=hidden_dim)
    if use_tensorops_v2 {
        let gate_off = read_off(gate_off_buf);
        let up_off = read_off(up_off_buf);
        // normed f32 → f16 (gate/up 공유, chain 1회). 이후 GEMM 들이 normed_f16_dev 를 device-direct.
        crate::compute::encode_cast_f32_to_f16(
            ctx,
            &enc,
            &carrier.normed_dev,
            &carrier.normed_f16_dev,
            &carrier.normed_elems_buf,
            m * hidden_dim,
        );
        crate::compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            &enc,
            gate_w_buf,
            gate_off,
            &carrier.normed_f16_dev,
            &carrier.gate_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
        crate::compute::encode_gemm_q4k_tensorops_v2(
            ctx,
            &enc,
            up_w_buf,
            up_off,
            &carrier.normed_f16_dev,
            &carrier.up_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
    } else if use_tensorops {
        let gate_off = read_off(gate_off_buf);
        let up_off = read_off(up_off_buf);
        crate::compute::encode_gemm_q4k_tensorops(
            ctx,
            &enc,
            gate_w_buf,
            gate_off,
            &carrier.normed_dev,
            &carrier.gate_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
        crate::compute::encode_gemm_q4k_tensorops(
            ctx,
            &enc,
            up_w_buf,
            up_off,
            &carrier.normed_dev,
            &carrier.up_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
    } else if naive {
        crate::compute::encode_gemm_q4k(
            ctx,
            &enc,
            gate_w_buf,
            &carrier.normed_dev,
            &carrier.gate_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            gate_off_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
        crate::compute::encode_gemm_q4k(
            ctx,
            &enc,
            up_w_buf,
            &carrier.normed_dev,
            &carrier.up_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            up_off_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
    } else {
        crate::compute::encode_gemm_q4k_shared(
            ctx,
            &enc,
            gate_w_buf,
            &carrier.normed_dev,
            &carrier.gate_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            gate_off_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
        crate::compute::encode_gemm_q4k_shared(
            ctx,
            &enc,
            up_w_buf,
            &carrier.normed_dev,
            &carrier.up_dev,
            &carrier.fdim_buf,
            &carrier.k_hidden_buf,
            up_off_buf,
            &carrier.m_buf,
            ffn_dim,
            m,
        );
    }
    if use_tensorops_v2 {
        crate::compute::encode_silu_mul_to_f16(
            ctx,
            &enc,
            &carrier.gate_dev,
            &carrier.up_dev,
            &carrier.act_f16_dev,
            &carrier.act_dim_buf,
            m * ffn_dim,
        );
    } else {
        // silu_mul: gate = silu(gate)*up (in-place, dim = m*ffn_dim)
        encode_silu_mul(
            ctx,
            &enc,
            &carrier.gate_dev,
            &carrier.up_dev,
            &carrier.act_dim_buf,
            m * ffn_dim,
        );
    }
    // down: gate[m,ffn] → down[m,hid]  (N=hidden_dim, K=ffn_dim)
    if use_tensorops_v2 {
        let down_off = read_off(down_off_buf);
        if down_is_q6k {
            crate::compute::encode_gemm_q6k_tensorops_v2(
                ctx,
                &enc,
                down_w_buf,
                down_off,
                &carrier.act_f16_dev,
                &carrier.down_dev,
                &carrier.hdim_buf,
                &carrier.k_ffn_buf,
                &carrier.m_buf,
                hidden_dim,
                m,
            );
        } else {
            crate::compute::encode_gemm_q4k_tensorops_v2(
                ctx,
                &enc,
                down_w_buf,
                down_off,
                &carrier.act_f16_dev,
                &carrier.down_dev,
                &carrier.hdim_buf,
                &carrier.k_ffn_buf,
                &carrier.m_buf,
                hidden_dim,
                m,
            );
        }
    } else if use_tensorops {
        let down_off = read_off(down_off_buf);
        if down_is_q6k {
            crate::compute::encode_gemm_q6k_tensorops(
                ctx,
                &enc,
                down_w_buf,
                down_off,
                &carrier.gate_dev,
                &carrier.down_dev,
                &carrier.hdim_buf,
                &carrier.k_ffn_buf,
                &carrier.m_buf,
                hidden_dim,
                m,
            );
        } else {
            crate::compute::encode_gemm_q4k_tensorops(
                ctx,
                &enc,
                down_w_buf,
                down_off,
                &carrier.gate_dev,
                &carrier.down_dev,
                &carrier.hdim_buf,
                &carrier.k_ffn_buf,
                &carrier.m_buf,
                hidden_dim,
                m,
            );
        }
    } else if down_is_q6k {
        if naive {
            crate::compute::encode_gemm_q6k(
                ctx,
                &enc,
                down_w_buf,
                &carrier.gate_dev,
                &carrier.down_dev,
                &carrier.hdim_buf,
                &carrier.k_ffn_buf,
                down_off_buf,
                &carrier.m_buf,
                hidden_dim,
                m,
            );
        } else {
            crate::compute::encode_gemm_q6k_shared(
                ctx,
                &enc,
                down_w_buf,
                &carrier.gate_dev,
                &carrier.down_dev,
                &carrier.hdim_buf,
                &carrier.k_ffn_buf,
                down_off_buf,
                &carrier.m_buf,
                hidden_dim,
                m,
            );
        }
    } else if naive {
        crate::compute::encode_gemm_q4k(
            ctx,
            &enc,
            down_w_buf,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.hdim_buf,
            &carrier.k_ffn_buf,
            down_off_buf,
            &carrier.m_buf,
            hidden_dim,
            m,
        );
    } else {
        crate::compute::encode_gemm_q4k_shared(
            ctx,
            &enc,
            down_w_buf,
            &carrier.gate_dev,
            &carrier.down_dev,
            &carrier.hdim_buf,
            &carrier.k_ffn_buf,
            down_off_buf,
            &carrier.m_buf,
            hidden_dim,
            m,
        );
    }

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // RNB_METAL_PREFILL_GPU_TIME=1: prefill FFN chain 의 발동 횟수 + command buffer GPU 실행시간 누적.
    // 발동 확인(call#) + per-layer GPU time. 한계: M5 dispatch-boundary 미지원이라 chain 내부
    // op(gate/up/silu/down) 분해 불가, command buffer 전체만. 매 호출(layer당 1회) 1줄, tail=총합.
    if std::env::var("RNB_METAL_PREFILL_GPU_TIME").as_deref() == Ok("1") {
        use objc2_metal::MTLCommandBuffer as _;
        let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
        PREFILL_FFN_GPU.with(|c| {
            let (cnt, sum) = c.get();
            let (cnt, sum) = (cnt + 1, sum + gpu_ms);
            c.set((cnt, sum));
            eprintln!(
                "[prefill-ffn-gpu] call#{cnt} gpu={gpu_ms:.3}ms cumulative={sum:.1}ms (m={m})"
            );
        });
    }

    readback(&carrier.down_dev, m * hidden_dim)
}

thread_local! {
    /// (호출 횟수, 누적 GPU ms). RNB_METAL_PREFILL_GPU_TIME 계측용.
    static PREFILL_FFN_GPU: Cell<(usize, f64)> = const { Cell::new((0, 0.0)) };
}

thread_local! {
    /// [count, gpu, wall, upload, encode, commit, wait, drv, sched] 누적 ms.
    static GLM_MOE_DECODE_PROFILE: Cell<[f64; 9]> = const { Cell::new([0.0; 9]) };
}

thread_local! {
    /// pm112 진단: MoE 스테이지별 GPU ms 누적 [gate_up_sparse, gate_up_shared,
    /// silu, down_sparse, down_shared, reduce, count].
    static GLM_MOE_STAGE_PROFILE: Cell<[f64; 7]> = const { Cell::new([0.0; 7]) };
}

/// pm112 진단: 현재 encoder 를 끊고 commit+wait 후 GPU 시간을 스테이지 슬롯에
/// 누적, 새 command buffer/encoder 로 교체한다. `RNB_METAL_GLM_MOE_STAGE_PROFILE=1`
/// 전용 — wall 이 크게 늘므로 측정 외 사용 금지.
fn glm_moe_stage_cut(
    ctx: &MetalContext,
    cmd: &mut Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: &mut Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    stage: usize,
) {
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0;
    GLM_MOE_STAGE_PROFILE.with(|acc| {
        let mut values = acc.get();
        values[stage] += gpu_ms;
        if stage == 5 {
            values[6] += 1.0;
            let n = values[6];
            if (n as usize) % 75 == 0 {
                eprintln!(
                    "[glm-moe-stage] calls={} gpu avg ms: gate_up_sparse={:.3} gate_up_shared={:.3} silu={:.3} down_sparse={:.3} down_shared={:.3} reduce={:.3}",
                    n as usize,
                    values[0] / n,
                    values[1] / n,
                    values[2] / n,
                    values[3] / n,
                    values[4] / n,
                    values[5] / n,
                );
            }
        }
        acc.set(values);
    });
    *cmd = ctx.queue.commandBuffer().expect("command buffer");
    *enc = cmd.computeCommandEncoder().expect("compute encoder");
}

// ---------------------------------------------------------------------------
// QKV projection chain — q/k/v 3 GEMV 를 단일 command buffer 로 (dispatch 3→1).
// pm8: 전체 decode device-resident chain 의 attention 조각. norm 은 앞 단계에서
// 끝나(scratch.norm_buf) chain 에 미포함 — q/k/v 는 norm 공유 + 서로 독립(병렬).
// ---------------------------------------------------------------------------

/// QKV projection 의 device-resident 버퍼 + scalar. (hidden_dim, q_out_dim, kv_dim)
/// 별 1회 alloc 후 재사용. `!Send+!Sync` 라 thread_local.
pub(crate) struct QkvCarrier {
    pub hidden_dim: usize,
    pub q_out_dim: usize,
    pub kv_dim: usize,
    pub norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub q_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub v_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub q_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // N=q_out_dim
    pub kv_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // N=kv_dim (k,v 공유)
    pub k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // K=hidden_dim
}

impl QkvCarrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        hidden_dim: usize,
        q_out_dim: usize,
        kv_dim: usize,
    ) -> Self {
        Self {
            hidden_dim,
            q_out_dim,
            kv_dim,
            norm_dev: empty_f32_buf(ctx, hidden_dim),
            q_dev: empty_f32_buf(ctx, q_out_dim),
            k_dev: empty_f32_buf(ctx, kv_dim),
            v_dev: empty_f32_buf(ctx, kv_dim),
            q_n_buf: u32_buf(ctx, q_out_dim as u32),
            kv_n_buf: u32_buf(ctx, kv_dim as u32),
            k_hidden_buf: u32_buf(ctx, hidden_dim as u32),
        }
    }

    /// norm host slice → norm_dev 업로드.
    fn upload_norm(&self, norm: &[f32]) {
        debug_assert_eq!(norm.len(), self.hidden_dim);
        let contents = self.norm_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                norm.as_ptr(),
                contents.as_ptr() as *mut f32,
                self.hidden_dim,
            );
        }
    }
}

/// QKV projection 한 token: q/k/v 3 GEMV 를 단일 command buffer 단일 encoder 로
/// (commit/wait 1). norm 은 모든 GEMV 공유(1회 업로드), q/k/v 출력은 서로 독립.
/// q/k/v weight buffer/offset(resident)은 caller 준비. 반환: (q, k, v).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qkv_chain_dispatch(
    ctx: &MetalContext,
    carrier: &QkvCarrier,
    norm_input: &[f32],
    q_w_buf: &ProtocolObject<dyn MTLBuffer>,
    q_off_buf: &ProtocolObject<dyn MTLBuffer>,
    k_w_buf: &ProtocolObject<dyn MTLBuffer>,
    k_off_buf: &ProtocolObject<dyn MTLBuffer>,
    v_w_buf: &ProtocolObject<dyn MTLBuffer>,
    v_off_buf: &ProtocolObject<dyn MTLBuffer>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    carrier.upload_norm(norm_input);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");

    // q: norm_dev → q_dev (N=q_out_dim, K=hidden_dim)
    crate::compute::encode_gemv_q4k(
        ctx,
        &enc,
        q_w_buf,
        &carrier.norm_dev,
        &carrier.q_dev,
        &carrier.q_n_buf,
        &carrier.k_hidden_buf,
        q_off_buf,
        carrier.q_out_dim,
    );
    // k: norm_dev → k_dev (N=kv_dim)
    crate::compute::encode_gemv_q4k(
        ctx,
        &enc,
        k_w_buf,
        &carrier.norm_dev,
        &carrier.k_dev,
        &carrier.kv_n_buf,
        &carrier.k_hidden_buf,
        k_off_buf,
        carrier.kv_dim,
    );
    // v: norm_dev → v_dev (N=kv_dim)
    crate::compute::encode_gemv_q4k(
        ctx,
        &enc,
        v_w_buf,
        &carrier.norm_dev,
        &carrier.v_dev,
        &carrier.kv_n_buf,
        &carrier.k_hidden_buf,
        v_off_buf,
        carrier.kv_dim,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    (
        readback(&carrier.q_dev, carrier.q_out_dim),
        readback(&carrier.k_dev, carrier.kv_dim),
        readback(&carrier.v_dev, carrier.kv_dim),
    )
}

// ---------------------------------------------------------------------------
// GDN in-projection chain — qkv + gate 2 GEMV 를 단일 command buffer 로.
// pm9: GDN layer(24개, 75%)의 앞부분. qkv/gate 는 norm 공유 + 서로 독립(병렬).
// alpha/beta/conv/scan 은 이후 단계 — 여기선 qkv+gate 만 (가장 작은 가치 단위).
// ---------------------------------------------------------------------------

/// GDN qkv+gate projection 의 device-resident 버퍼 + scalar.
/// (hidden_dim, qkv_dim=conv_channels, gate_dim=d_inner) 별 1회 alloc 후 재사용.
pub(crate) struct GdnInprojCarrier {
    pub hidden_dim: usize,
    pub qkv_dim: usize,
    pub gate_dim: usize,
    pub norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub qkv_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub qkv_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub gate_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k_hidden_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl GdnInprojCarrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        hidden_dim: usize,
        qkv_dim: usize,
        gate_dim: usize,
    ) -> Self {
        Self {
            hidden_dim,
            qkv_dim,
            gate_dim,
            norm_dev: empty_f32_buf(ctx, hidden_dim),
            qkv_dev: empty_f32_buf(ctx, qkv_dim),
            gate_dev: empty_f32_buf(ctx, gate_dim),
            qkv_n_buf: u32_buf(ctx, qkv_dim as u32),
            gate_n_buf: u32_buf(ctx, gate_dim as u32),
            k_hidden_buf: u32_buf(ctx, hidden_dim as u32),
        }
    }

    fn upload_norm(&self, norm: &[f32]) {
        debug_assert_eq!(norm.len(), self.hidden_dim);
        let contents = self.norm_dev.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                norm.as_ptr(),
                contents.as_ptr() as *mut f32,
                self.hidden_dim,
            );
        }
    }
}

/// GDN qkv+gate 한 token: 2 GEMV 를 단일 command buffer 단일 encoder 로(commit/wait 1).
/// norm 공유(1회 업로드), qkv/gate 출력 독립. 반환: (qkv, gate).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_inproj_chain_dispatch(
    ctx: &MetalContext,
    carrier: &GdnInprojCarrier,
    norm_input: &[f32],
    qkv_w_buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_off_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_w_buf: &ProtocolObject<dyn MTLBuffer>,
    gate_off_buf: &ProtocolObject<dyn MTLBuffer>,
) -> (Vec<f32>, Vec<f32>) {
    carrier.upload_norm(norm_input);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");

    crate::compute::encode_gemv_q4k(
        ctx,
        &enc,
        qkv_w_buf,
        &carrier.norm_dev,
        &carrier.qkv_dev,
        &carrier.qkv_n_buf,
        &carrier.k_hidden_buf,
        qkv_off_buf,
        carrier.qkv_dim,
    );
    crate::compute::encode_gemv_q4k(
        ctx,
        &enc,
        gate_w_buf,
        &carrier.norm_dev,
        &carrier.gate_dev,
        &carrier.gate_n_buf,
        &carrier.k_hidden_buf,
        gate_off_buf,
        carrier.gate_dim,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    (
        readback(&carrier.qkv_dev, carrier.qkv_dim),
        readback(&carrier.gate_dev, carrier.gate_dim),
    )
}

// ---------------------------------------------------------------------------
// attention O chain — o_proj + residual 을 단일 command buffer 로.
// pm10: attention compute(KV, host) 후 단계. o_proj GEMV → residual(hidden+=proj).
// ffn chain 의 down+residual 패턴과 동일. non-gemma(qwen) 한정(gemma post-norm 제외).
// ---------------------------------------------------------------------------

/// attention O projection + residual 의 device-resident 버퍼.
/// (hidden_dim, q_dim) 별 1회 alloc 후 재사용.
pub(crate) struct OChainCarrier {
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub attn_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub hidden_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k_q_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl OChainCarrier {
    pub(crate) fn new(ctx: &MetalContext, hidden_dim: usize, q_dim: usize) -> Self {
        Self {
            hidden_dim,
            q_dim,
            attn_dev: empty_f32_buf(ctx, q_dim),
            hidden_dev: empty_f32_buf(ctx, hidden_dim),
            proj_dev: empty_f32_buf(ctx, hidden_dim),
            hidden_n_buf: u32_buf(ctx, hidden_dim as u32),
            k_q_buf: u32_buf(ctx, q_dim as u32),
        }
    }

    fn upload(&self, attn_out: &[f32], hidden: &[f32]) {
        debug_assert_eq!(attn_out.len(), self.q_dim);
        debug_assert_eq!(hidden.len(), self.hidden_dim);
        unsafe {
            std::ptr::copy_nonoverlapping(
                attn_out.as_ptr(),
                self.attn_dev.contents().as_ptr() as *mut f32,
                self.q_dim,
            );
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                self.hidden_dev.contents().as_ptr() as *mut f32,
                self.hidden_dim,
            );
        }
    }
}

/// attention O chain 한 token: o_proj GEMV(attn→proj) + residual(hidden+=proj)
/// 단일 command buffer. 반환: residual 적용된 hidden(hidden_dim).
pub(crate) fn o_chain_dispatch(
    ctx: &MetalContext,
    carrier: &OChainCarrier,
    attn_out: &[f32],
    hidden: &[f32],
    o_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_off_buf: &ProtocolObject<dyn MTLBuffer>,
) -> Vec<f32> {
    carrier.upload(attn_out, hidden);

    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");

    // o_proj: attn_dev → proj_dev (N=hidden_dim, K=q_dim)
    crate::compute::encode_gemv_q4k(
        ctx,
        &enc,
        o_w_buf,
        &carrier.attn_dev,
        &carrier.proj_dev,
        &carrier.hidden_n_buf,
        &carrier.k_q_buf,
        o_off_buf,
        carrier.hidden_dim,
    );
    // residual: hidden_dev += proj_dev (hazard tracking 자동 직렬화)
    encode_residual_add(
        ctx,
        &enc,
        &carrier.hidden_dev,
        &carrier.proj_dev,
        &carrier.hidden_n_buf,
        carrier.hidden_dim,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    readback(&carrier.hidden_dev, carrier.hidden_dim)
}

#[cfg(test)]
mod qwen_moe_id_plan_tests {
    use super::*;

    #[test]
    fn qwen_moe_llama_map0_cpu_builds_expert_major_valid_prefixes() {
        let selected_experts = [2, 0, 3, 1];
        let (tpe, ids) =
            qwen_moe_llama_map0_cpu(&selected_experts, 2, 2, 4).expect("valid route map");

        assert_eq!(tpe, vec![1, 1, 1, 1]);
        assert_eq!(&ids[0..1], &[1]);
        assert_eq!(&ids[2..3], &[3]);
        assert_eq!(&ids[4..5], &[0]);
        assert_eq!(&ids[6..7], &[2]);
    }

    #[test]
    fn qwen_moe_llama_map0_cpu_rejects_invalid_layout() {
        assert_eq!(
            qwen_moe_llama_map0_cpu(&[0, 1, 2], 2, 2, 4),
            Err(QwenMoeLlamaIdError::InvalidSelectedExpertLayout)
        );
        assert_eq!(
            qwen_moe_llama_map0_cpu(&[0, 4], 1, 2, 4),
            Err(QwenMoeLlamaIdError::ExpertIdOutOfRange)
        );
        assert_eq!(
            qwen_moe_llama_map0_cpu(&[1, 1], 1, 2, 4),
            Err(QwenMoeLlamaIdError::InvalidSelectedExpertLayout)
        );
    }

    #[test]
    fn qwen_moe_llama_id_plan_uses_checked_aligned_component_sizes() {
        let preflight =
            qwen_moe_llama_id_preflight(true, true, 33, 5, 3, 7, 65, usize::MAX).unwrap();
        let QwenMoeLlamaIdPreflight::Run(plan) = preflight else {
            panic!("supported shape should run");
        };
        let slots = 33usize * 3;
        let aligned = |bytes: usize| {
            bytes.div_ceil(QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT) * QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT
        };

        assert_eq!(plan.n_tokens, 33);
        assert_eq!(plan.n_expert, 5);
        assert_eq!(plan.n_expert_used, 3);
        assert_eq!(plan.hidden_dim, 7);
        assert_eq!(plan.ffn_dim, 65);
        assert_eq!(plan.gate_bytes, aligned(slots * 65 * 4));
        assert_eq!(plan.up_bytes, aligned(slots * 65 * 4));
        assert_eq!(plan.down_rank_bytes, aligned(slots * 7 * 4));
        assert_eq!(plan.selected_expert_bytes, aligned(slots * 4));
        assert_eq!(plan.route_weight_bytes, aligned(slots * 4));
        assert_eq!(plan.shared_route_weight_bytes, aligned(33 * 4));
        assert_eq!(plan.tpe_bytes, aligned(5 * 4));
        assert_eq!(plan.ids_bytes, aligned(5 * 33 * 4));
        assert_eq!(plan.q8_bytes, aligned(slots * 292));
        assert_eq!(plan.sparse_moe_out_bytes, aligned(33 * 7 * 4));
        assert_eq!(plan.shared_out_bytes, aligned(33 * 7 * 4));
        assert_eq!(
            plan.total_bytes,
            plan.gate_bytes
                + plan.up_bytes
                + plan.down_rank_bytes
                + plan.selected_expert_bytes
                + plan.route_weight_bytes
                + plan.shared_route_weight_bytes
                + plan.tpe_bytes
                + plan.ids_bytes
                + plan.q8_bytes
                + plan.sparse_moe_out_bytes
                + plan.shared_out_bytes
        );
    }

    #[test]
    fn qwen_moe_llama_id_plan_rejects_shape_and_arithmetic_overflow() {
        assert_eq!(
            qwen_moe_llama_id_preflight(true, true, 1, 1, 0, 1, 1, usize::MAX),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        assert_eq!(
            qwen_moe_llama_id_preflight(true, true, 1, 32, 9, 1, 1, usize::MAX),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        assert_eq!(
            qwen_moe_llama_id_preflight(true, true, usize::MAX, 1, 2, 1, 1, usize::MAX),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        assert_eq!(
            qwen_moe_llama_id_checked_aligned_bytes(&[usize::MAX, 2], 4),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        assert_eq!(
            qwen_moe_llama_id_checked_aligned_bytes(&[usize::MAX / 4], 4),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        assert_eq!(
            qwen_moe_llama_id_checked_total(&[usize::MAX - 255, 256]),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
    }

    #[test]
    fn qwen_moe_llama_id_plan_limits_fallback_to_presubmit_capabilities_and_budget() {
        assert_eq!(
            qwen_moe_llama_id_preflight(false, true, 1, 1, 1, 1, 1, usize::MAX),
            Ok(QwenMoeLlamaIdPreflight::Fallback(
                QwenMoeLlamaIdFallbackReason::PipelineUnavailable
            ))
        );
        assert_eq!(
            qwen_moe_llama_id_preflight(true, false, 1, 1, 1, 1, 1, usize::MAX),
            Ok(QwenMoeLlamaIdPreflight::Fallback(
                QwenMoeLlamaIdFallbackReason::UnsupportedQuant
            ))
        );
        assert_eq!(
            qwen_moe_llama_id_preflight(true, true, 1, 1, 1, 1, 1, 0),
            Ok(QwenMoeLlamaIdPreflight::Fallback(
                QwenMoeLlamaIdFallbackReason::ScratchOverBudget
            ))
        );

        let execution_errors = [
            QwenMoeLlamaIdError::InvalidShape,
            QwenMoeLlamaIdError::InvalidExpertArena,
            QwenMoeLlamaIdError::ExpertArenaMismatch,
            QwenMoeLlamaIdError::ExpertIdOutOfRange,
            QwenMoeLlamaIdError::TokenIdOutOfRange,
            QwenMoeLlamaIdError::InvalidSelectedExpertLayout,
            QwenMoeLlamaIdError::DispatchGridOverflow,
            QwenMoeLlamaIdError::AllocationFailed,
            QwenMoeLlamaIdError::DeviceMapFailed,
            QwenMoeLlamaIdError::CommandBufferFailed,
        ];
        assert_eq!(execution_errors.len(), 10);
    }

    fn test_shape() -> QwenMoeIdPrimitiveShape {
        QwenMoeIdPrimitiveShape {
            tile_slots: 64,
            ffn_tile: 256,
            hidden_tile: 64,
            ffn_tiles_per_pass: 2,
        }
    }

    #[test]
    fn qwen_moe_id_plan_preserves_presorted_slots_and_weights() {
        let expert_ids = [0, 0, 2, 2, 2, 5];
        let token_ids = [3, 1, 3, 2, 0, 1];
        let route_weights = [0.70, 0.20, 0.10, 0.30, 0.40, 0.90];

        let plan =
            build_qwen_moe_id_execution_plan(&expert_ids, &token_ids, &route_weights, 6, 4, 64)
                .expect("valid plan");

        assert_eq!(plan.expert_offsets, vec![0, 0, 2, 0, 0, 5]);
        assert_eq!(plan.expert_counts, vec![2, 0, 3, 0, 0, 1]);
        assert_eq!(plan.block_experts, vec![0, 2, 5]);
        assert_eq!(plan.block_local0, vec![0, 0, 0]);
        assert_eq!(plan.token_ids, token_ids);
        assert_eq!(plan.route_weights, route_weights);
    }

    #[test]
    fn qwen_moe_v3_plan_sorts_token_major_slots_and_preserves_rank() {
        let expert_ids = [2, 0, 1, 2, 0, 1];
        let token_ids = [0, 0, 0, 1, 1, 1];
        let route_weights = [0.6, 0.3, 0.1, 0.5, 0.4, 0.1];

        let plan = build_qwen_moe_mul_mm_id_v3_plan_token_major(
            &expert_ids,
            &token_ids,
            &route_weights,
            3,
            2,
            3,
            64,
        )
        .expect("v3 plan");

        assert_eq!(plan.expert_offsets, vec![0, 2, 4]);
        assert_eq!(plan.expert_counts, vec![2, 2, 2]);
        assert_eq!(plan.block_experts, vec![0, 1, 2]);
        assert_eq!(plan.block_local0, vec![0, 0, 0]);
        assert_eq!(plan.token_ids_sorted, vec![0, 1, 0, 1, 0, 1]);
        assert_eq!(plan.rank_ids_sorted, vec![1, 1, 2, 2, 0, 0]);
        assert_eq!(plan.dst_slots_sorted, vec![1, 4, 2, 5, 0, 3]);
        assert_eq!(plan.route_weights_token_rank, route_weights);
    }

    #[test]
    fn qwen_moe_v3_plan_rejects_expert_major_without_rank_metadata() {
        let err = build_qwen_moe_mul_mm_id_v3_plan_token_major(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[0.1, 0.2, 0.3, 0.4],
            2,
            2,
            2,
            64,
        )
        .unwrap_err();
        assert_eq!(err, QwenMoeIdPreflightReason::RankCountMismatch);
    }

    #[test]
    fn qwen_moe_v3_scratch_budget_uses_checked_component_sizes() {
        let budget =
            QwenMoeMulMmIdV3Scratch::checked(3648, 2048, 512, 8, 128, 128, 128 * 1024 * 1024)
                .expect("within budget");
        let slots = 512 * 8;
        assert_eq!(budget.gate_bytes, slots * 3648 * 2);
        assert_eq!(budget.up_bytes, slots * 3648 * 2);
        assert_eq!(budget.down_bytes, slots * 2048 * 4);
        assert!(budget.metadata_bytes > 0);
        assert_eq!(
            budget.total_bytes,
            budget.gate_bytes + budget.up_bytes + budget.down_bytes + budget.metadata_bytes
        );
        assert!(budget.total_bytes <= 128 * 1024 * 1024);
    }

    #[test]
    fn qwen_moe_v3_scratch_budget_rejects_over_budget() {
        let err = QwenMoeMulMmIdV3Scratch::checked(3648, 2048, 512, 8, 128, 128, 64 * 1024 * 1024)
            .unwrap_err();
        assert_eq!(err, QwenMoeIdPreflightReason::ScratchOverBudget);
    }

    #[test]
    fn qwen_moe_v3_dims_require_qk_superblock_aligned_k() {
        assert!(qwen_moe_mulmmid_v3_dims_supported(2048, 7680));
        assert!(!qwen_moe_mulmmid_v3_dims_supported(0, 7680));
        assert!(!qwen_moe_mulmmid_v3_dims_supported(2048, 0));
        assert!(!qwen_moe_mulmmid_v3_dims_supported(2049, 7680));
        assert!(!qwen_moe_mulmmid_v3_dims_supported(2048, 7681));
    }

    #[test]
    fn qwen_moe_v4_plan_sorts_by_expert_rank_token_and_builds_gate_blocks() {
        let expert_ids = [2, 0, 1, 2, 2, 0, 1, 2];
        let token_ids = [0, 0, 1, 1, 2, 2, 3, 3];
        let route_weights = [0.60, 0.30, 0.50, 0.40, 0.70, 0.20, 0.80, 0.10];

        let plan = build_qwen_moe_mul_mm_id_v4_plan_token_major(
            &expert_ids,
            &token_ids,
            &route_weights,
            3,
            4,
            2,
            64,
        )
        .expect("v4 plan");

        assert_eq!(plan.expert_offsets, vec![0, 2, 4]);
        assert_eq!(plan.expert_counts, vec![2, 2, 4]);
        assert_eq!(plan.id_block_experts, vec![0, 1, 2]);
        assert_eq!(plan.id_block_local0, vec![0, 0, 0]);
        assert_eq!(plan.token_ids_sorted, vec![0, 2, 1, 3, 0, 2, 1, 3]);
        assert_eq!(plan.rank_ids_sorted, vec![1, 1, 0, 0, 0, 0, 1, 1]);
        assert_eq!(plan.expert_rank_offsets, vec![0, 0, 2, 4, 4, 6]);
        assert_eq!(plan.expert_rank_counts, vec![0, 2, 2, 0, 2, 2]);
        assert_eq!(plan.rank_block_offsets, vec![0, 2]);
        assert_eq!(plan.rank_block_counts, vec![2, 2]);
        assert_eq!(plan.rank_block_experts, vec![1, 2, 0, 2]);
        assert_eq!(plan.rank_block_local0, vec![0, 0, 0, 0]);
    }

    #[test]
    fn qwen_moe_v4_plan_uses_rank_local_block_offsets_for_large_rank_group() {
        let seq_len = 130usize;
        let mut expert_ids = Vec::with_capacity(seq_len * 2);
        let mut token_ids = Vec::with_capacity(seq_len * 2);
        let mut route_weights = Vec::with_capacity(seq_len * 2);
        for token in 0..seq_len {
            expert_ids.extend([1, 2]);
            token_ids.extend([token as u32, token as u32]);
            route_weights.extend([0.6, 0.4]);
        }

        let plan = build_qwen_moe_mul_mm_id_v4_plan_token_major(
            &expert_ids,
            &token_ids,
            &route_weights,
            3,
            seq_len,
            2,
            64,
        )
        .expect("v4 plan");

        assert_eq!(plan.expert_rank_counts[2], 130);
        assert_eq!(plan.expert_rank_counts[5], 130);
        assert_eq!(plan.rank_block_offsets, vec![0, 3]);
        assert_eq!(plan.rank_block_counts, vec![3, 3]);
        assert_eq!(plan.rank_block_experts, vec![1, 1, 1, 2, 2, 2]);
        assert_eq!(plan.rank_block_local0, vec![0, 64, 128, 0, 64, 128]);
    }

    #[test]
    fn qwen_moe_v4_plan_rejects_duplicate_expert_token_across_ranks() {
        let err = build_qwen_moe_mul_mm_id_v4_plan_token_major(
            &[2, 2, 2, 0],
            &[0, 0, 1, 1],
            &[0.6, 0.4, 0.7, 0.3],
            3,
            2,
            2,
            64,
        )
        .expect_err("duplicate expert/token should fail");
        assert_eq!(err, QwenMoeIdPreflightReason::DuplicateTokenInExpertBlock);
    }

    #[test]
    fn qwen_moe_v4_scratch_budget_excludes_down_token_rank_scratch() {
        let budget = QwenMoeMulMmIdV4Scratch::checked(512, 256, 128, 8, 128, 64, 256 * 1024 * 1024)
            .expect("within budget");
        let slots = 128 * 8;
        assert_eq!(budget.gate_bytes, slots * 512 * 2);
        assert_eq!(budget.up_bytes, slots * 512 * 2);
        assert_eq!(budget.down_bytes, 0);
        assert!(budget.metadata_bytes > 0);
        assert_eq!(
            budget.total_bytes,
            budget.gate_bytes + budget.up_bytes + budget.metadata_bytes
        );
    }

    #[test]
    fn qwen_moe_v4_dims_require_qk_superblock_aligned_k() {
        assert!(qwen_moe_mulmmid_v4_dims_supported(2048, 7680));
        assert!(!qwen_moe_mulmmid_v4_dims_supported(2049, 7680));
        assert!(!qwen_moe_mulmmid_v4_dims_supported(2048, 7681));
    }

    #[test]
    fn qwen_moe_raw_id_map_uses_seq_len_stride_not_route_slots() {
        let map = build_qwen_moe_raw_id_map_from_sorted(&[0, 0, 2, 2, 3], &[2, 0, 1, 2, 1], 4, 3)
            .expect("raw id map");

        assert_eq!(map.expert_counts, vec![2, 0, 2, 1]);
        assert_eq!(map.route_slot_ids.len(), 4 * 3);
        assert_eq!(map.route_slot_ids[0..3], [0, 1, u32::MAX]);
        assert_eq!(map.route_slot_ids[3..6], [u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(map.route_slot_ids[6..9], [2, 3, u32::MAX]);
        assert_eq!(map.route_slot_ids[9..12], [4, u32::MAX, u32::MAX]);
    }

    #[test]
    fn qwen_moe_raw_id_map_rejects_unsorted_and_duplicate_token_per_expert() {
        let unsorted = build_qwen_moe_raw_id_map_from_sorted(&[1, 0], &[0, 1], 2, 2)
            .expect_err("unsorted expert ids");
        assert_eq!(unsorted, QwenMoeIdPreflightReason::UnsortedExpertIds);

        let duplicate = build_qwen_moe_raw_id_map_from_sorted(&[0, 0], &[1, 1], 1, 2)
            .expect_err("duplicate token in expert");
        assert_eq!(
            duplicate,
            QwenMoeIdPreflightReason::DuplicateTokenInExpertBlock
        );
    }

    #[test]
    fn qwen_moe_id_plan_rejects_invalid_inputs_with_reasons() {
        assert_eq!(
            build_qwen_moe_id_execution_plan(&[1, 0], &[0, 1], &[0.5, 0.5], 2, 2, 64).unwrap_err(),
            QwenMoeIdPreflightReason::UnsortedExpertIds
        );
        assert_eq!(
            build_qwen_moe_id_execution_plan(&[2], &[0], &[1.0], 2, 1, 64).unwrap_err(),
            QwenMoeIdPreflightReason::ExpertIdOutOfRange
        );
        assert_eq!(
            build_qwen_moe_id_execution_plan(&[0], &[1], &[1.0], 1, 1, 64).unwrap_err(),
            QwenMoeIdPreflightReason::TokenIdOutOfRange
        );
        assert_eq!(
            build_qwen_moe_id_execution_plan(&[0, 0], &[1, 1], &[0.5, 0.5], 1, 2, 64).unwrap_err(),
            QwenMoeIdPreflightReason::DuplicateTokenInExpertBlock
        );
        assert!(
            build_qwen_moe_id_execution_plan(&[0, 1], &[1, 1], &[0.5, 0.5], 2, 2, 64).is_ok(),
            "same token may appear in different expert groups"
        );
        assert_eq!(
            build_qwen_moe_id_execution_plan(&[0], &[0], &[1.0], 1, 1, 0).unwrap_err(),
            QwenMoeIdPreflightReason::InvalidExpertArena
        );
    }

    #[test]
    fn qwen_moe_id_primitive_carrier_scratch_is_bounded_by_tiles_not_slots_ffn() {
        let shape = QwenMoeIdPrimitiveShape {
            tile_slots: 64,
            ffn_tile: 64,
            hidden_tile: 64,
            ffn_tiles_per_pass: 192,
        };
        let scratch = shape.scratch_bytes();
        let forbidden_full_gate_up = 512usize * 12_288usize * 2usize * std::mem::size_of::<f32>();
        assert!(scratch < forbidden_full_gate_up / 8);
        assert_eq!(
            scratch,
            (64 * 64 * 3 + 192 * 64 * 64) * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn qwen_moe_id_primitive_dispatch_budget_rejects_hidden_ffn_cross_product() {
        let shape = QwenMoeIdPrimitiveShape {
            tile_slots: QWEN_MOE_ID_PRIMITIVE_TILE_SLOTS,
            ffn_tile: QWEN_MOE_ID_PRIMITIVE_FFN_TILE,
            hidden_tile: QWEN_MOE_ID_PRIMITIVE_HIDDEN_TILE,
            ffn_tiles_per_pass: 30,
        };

        let predicted = qwen_moe_id_primitive_predicted_dispatch_count(55, 2048, shape)
            .expect("predicted dispatch count");
        let budget =
            qwen_moe_id_primitive_dispatch_budget(55, 2048, shape).expect("dispatch budget");

        assert_eq!(predicted, 55 * 32 * (30 * 3 + 1));
        assert_eq!(budget, 55 * (30 * 3 + 32 + 1) * 4);
        assert!(predicted > budget);
    }

    #[test]
    fn qwen_moe_id_primitive_dispatch_budget_allows_small_oracle_shape() {
        let shape = QwenMoeIdPrimitiveShape {
            tile_slots: QWEN_MOE_ID_PRIMITIVE_TILE_SLOTS,
            ffn_tile: QWEN_MOE_ID_PRIMITIVE_FFN_TILE,
            hidden_tile: QWEN_MOE_ID_PRIMITIVE_HIDDEN_TILE,
            ffn_tiles_per_pass: 2,
        };

        let predicted = qwen_moe_id_primitive_predicted_dispatch_count(2, 64, shape)
            .expect("predicted dispatch count");
        let budget = qwen_moe_id_primitive_dispatch_budget(2, 64, shape).expect("dispatch budget");

        assert_eq!(predicted, 2 * (2 * 3 + 1));
        assert!(predicted <= budget);
    }

    #[test]
    fn qwen_moe_id_primitive_preflight_reports_pipeline_unavailable() {
        let reason = preflight_qwen_moe_id_primitive_inputs(
            false,
            true,
            144,
            144,
            210,
            144,
            144,
            210,
            &[0],
            &[0],
            &[1.0],
            1,
            96,
            512,
            64,
            test_shape(),
            usize::MAX / 2,
        )
        .unwrap_err();
        assert_eq!(reason.as_str(), "pipeline_unavailable");
    }

    #[test]
    fn qwen_moe_id_primitive_preflight_reports_arena_errors() {
        let invalid = preflight_qwen_moe_id_primitive_inputs(
            true,
            true,
            145,
            144,
            210,
            144,
            144,
            210,
            &[0],
            &[0],
            &[1.0],
            1,
            96,
            512,
            64,
            test_shape(),
            usize::MAX / 2,
        )
        .unwrap_err();
        assert_eq!(invalid.as_str(), "invalid_expert_arena");

        let mismatch = preflight_qwen_moe_id_primitive_inputs(
            true,
            true,
            288,
            144,
            210,
            144,
            144,
            210,
            &[0],
            &[0],
            &[1.0],
            1,
            96,
            512,
            64,
            test_shape(),
            usize::MAX / 2,
        )
        .unwrap_err();
        assert_eq!(mismatch.as_str(), "expert_arena_mismatch");
    }

    #[test]
    fn qwen_moe_id_primitive_preflight_reports_sorted_and_duplicate_errors() {
        let unsorted = preflight_qwen_moe_id_primitive_inputs(
            true,
            true,
            288,
            288,
            420,
            144,
            144,
            210,
            &[1, 0],
            &[0, 1],
            &[0.5, 0.5],
            2,
            96,
            512,
            64,
            test_shape(),
            usize::MAX / 2,
        )
        .unwrap_err();
        assert_eq!(unsorted.as_str(), "unsorted_expert_ids");

        let duplicate = preflight_qwen_moe_id_primitive_inputs(
            true,
            true,
            144,
            144,
            210,
            144,
            144,
            210,
            &[0, 0],
            &[0, 0],
            &[0.5, 0.5],
            1,
            96,
            512,
            64,
            test_shape(),
            usize::MAX / 2,
        )
        .unwrap_err();
        assert_eq!(duplicate.as_str(), "duplicate_token_in_expert_block");
    }

    #[test]
    fn qwen_moe_id_primitive_preflight_rejects_non_q6_down() {
        let reason = preflight_qwen_moe_id_primitive_inputs(
            true,
            false,
            144,
            144,
            210,
            144,
            144,
            210,
            &[0],
            &[0],
            &[1.0],
            1,
            96,
            512,
            64,
            test_shape(),
            usize::MAX / 2,
        )
        .unwrap_err();
        assert_eq!(reason.as_str(), "unsupported_down_quant");
    }

    #[test]
    fn qwen_moe_id_primitive_preflight_reports_partial_budget() {
        let shape = test_shape();
        let reason = preflight_qwen_moe_id_primitive_inputs(
            true,
            true,
            144,
            144,
            210,
            144,
            144,
            210,
            &[0],
            &[0],
            &[1.0],
            1,
            96,
            512,
            64,
            shape,
            shape.partial_bytes() - 1,
        )
        .unwrap_err();
        assert_eq!(reason.as_str(), "partial_buffer_over_budget");
    }

    #[test]
    fn qwen_moe_id_primitive_preflight_rejects_shape_mismatch() {
        let mut shape = test_shape();
        shape.ffn_tiles_per_pass = 1;
        let reason = preflight_qwen_moe_id_primitive_inputs(
            true,
            true,
            144,
            144,
            210,
            144,
            144,
            210,
            &[0],
            &[0],
            &[1.0],
            1,
            96,
            512,
            64,
            shape,
            usize::MAX / 2,
        )
        .unwrap_err();
        assert_eq!(reason.as_str(), "invalid_primitive_shape");
    }

    fn qwen_moe_llama_carrier_test_plan() -> QwenMoeLlamaIdPlan {
        let QwenMoeLlamaIdPreflight::Run(plan) =
            qwen_moe_llama_id_preflight(true, true, 3, 5, 2, 256, 512, usize::MAX)
                .expect("valid carrier plan")
        else {
            panic!("valid carrier shape must run");
        };
        plan
    }

    #[test]
    fn qwen_moe_llama_carrier_checked_bytes_match_unique_live_allocations() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct FakeAllocation {
            id: usize,
            bytes: usize,
            live: Rc<Cell<usize>>,
        }

        impl Drop for FakeAllocation {
            fn drop(&mut self) {
                self.live.set(self.live.get() - 1);
            }
        }

        let plan = qwen_moe_llama_carrier_test_plan();
        let selected = [0, 2, 1, 3, 4, 0];
        let route_weights = [0.7, 0.3, 0.6, 0.4, 0.8, 0.2];
        let shared_route_weights = [0.25, 0.5, 0.75];
        let live = Rc::new(Cell::new(0));
        let mut next_id = 0usize;
        let allocations = qwen_moe_llama_carrier_allocate_with(
            plan,
            &selected,
            &route_weights,
            &shared_route_weights,
            |bytes| {
                let id = next_id;
                next_id += 1;
                live.set(live.get() + 1);
                Some(FakeAllocation {
                    id,
                    bytes,
                    live: Rc::clone(&live),
                })
            },
        )
        .expect("checked fake allocations");

        assert_eq!(allocations.len(), 12);
        assert_eq!(live.get(), 12);
        assert_eq!(
            allocations
                .iter()
                .map(|allocation| allocation.id)
                .collect::<Vec<_>>(),
            (0..12).collect::<Vec<_>>()
        );
        assert_eq!(
            allocations
                .iter()
                .map(|allocation| allocation.bytes)
                .collect::<Vec<_>>(),
            qwen_moe_llama_carrier_allocation_bytes(plan)
        );
        assert_eq!(
            allocations
                .iter()
                .map(|allocation| allocation.bytes)
                .sum::<usize>(),
            plan.total_bytes
        );
        assert!(allocations
            .iter()
            .all(|allocation| { allocation.bytes % QWEN_MOE_LLAMA_ID_BUFFER_ALIGNMENT == 0 }));

        drop(allocations);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn qwen_moe_llama_carrier_allocation_failure_is_error_and_drops_partial_allocations() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct FakeAllocation(Rc<Cell<usize>>);

        impl Drop for FakeAllocation {
            fn drop(&mut self) {
                self.0.set(self.0.get() - 1);
            }
        }

        let plan = qwen_moe_llama_carrier_test_plan();
        let selected = [0, 2, 1, 3, 4, 0];
        let route_weights = [0.7, 0.3, 0.6, 0.4, 0.8, 0.2];
        let shared_route_weights = [0.25, 0.5, 0.75];
        let live = Rc::new(Cell::new(0));
        let calls = Cell::new(0usize);
        let result = qwen_moe_llama_carrier_allocate_with(
            plan,
            &selected,
            &route_weights,
            &shared_route_weights,
            |_| {
                let call = calls.get();
                calls.set(call + 1);
                if call == 4 {
                    None
                } else {
                    live.set(live.get() + 1);
                    Some(FakeAllocation(Rc::clone(&live)))
                }
            },
        );

        assert!(matches!(result, Err(QwenMoeLlamaIdError::AllocationFailed)));
        assert_eq!(calls.get(), 5);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn qwen_moe_llama_carrier_zero_rank_and_invalid_routes_reject_before_allocation() {
        let valid_plan = qwen_moe_llama_carrier_test_plan();
        let calls = Cell::new(0usize);
        let mut zero_rank_plan = valid_plan;
        zero_rank_plan.n_expert_used = 0;
        assert_eq!(
            qwen_moe_llama_carrier_allocate_with(zero_rank_plan, &[], &[], &[0.0; 3], |_| {
                calls.set(calls.get() + 1);
                Some(())
            },),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        assert_eq!(calls.get(), 0);

        let route_weights = [0.7, 0.3, 0.6, 0.4, 0.8, 0.2];
        let shared_route_weights = [0.25, 0.5, 0.75];
        for (selected, expected) in [
            (
                [0, 0, 1, 3, 4, 0],
                QwenMoeLlamaIdError::InvalidSelectedExpertLayout,
            ),
            ([0, 5, 1, 3, 4, 0], QwenMoeLlamaIdError::ExpertIdOutOfRange),
        ] {
            assert_eq!(
                qwen_moe_llama_carrier_allocate_with(
                    valid_plan,
                    &selected,
                    &route_weights,
                    &shared_route_weights,
                    |_| {
                        calls.set(calls.get() + 1);
                        Some(())
                    },
                ),
                Err(expected)
            );
            assert_eq!(calls.get(), 0);
        }
    }

    #[test]
    fn qwen_moe_llama_carrier_budget_fallback_happens_before_any_allocation_plan() {
        let plan = qwen_moe_llama_carrier_test_plan();
        assert_eq!(
            qwen_moe_llama_id_preflight(
                true,
                true,
                plan.n_tokens,
                plan.n_expert,
                plan.n_expert_used,
                plan.hidden_dim,
                plan.ffn_dim,
                plan.total_bytes - 1,
            ),
            Ok(QwenMoeLlamaIdPreflight::Fallback(
                QwenMoeLlamaIdFallbackReason::ScratchOverBudget
            ))
        );
    }

    #[test]
    fn qwen_moe_device_route_top1_reserves_shared_f16_arena_and_budget_falls_back() {
        const N_TOKENS: usize = 128;
        const N_EXPERT: usize = 4;
        const N_EXPERT_USED: usize = 1;
        const HIDDEN_DIM: usize = 256;
        const FFN_DIM: usize = 1024;

        let generic = match qwen_moe_llama_id_preflight(
            true,
            true,
            N_TOKENS,
            N_EXPERT,
            N_EXPERT_USED,
            HIDDEN_DIM,
            FFN_DIM,
            usize::MAX,
        )
        .expect("generic top-1 plan")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("unexpected generic fallback: {reason:?}")
            }
        };
        let device_route = match qwen_moe_prefill_device_route_preflight(
            true,
            true,
            N_TOKENS,
            N_EXPERT,
            N_EXPERT_USED,
            HIDDEN_DIM,
            FFN_DIM,
            usize::MAX,
        )
        .expect("device-route top-1 plan")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("unexpected device-route fallback: {reason:?}")
            }
        };
        let expected_shared_arena = N_TOKENS * (HIDDEN_DIM + FFN_DIM) * std::mem::size_of::<u16>();
        assert_eq!(device_route.down_rank_bytes, expected_shared_arena);
        assert_eq!(
            device_route.total_bytes,
            generic.total_bytes - generic.down_rank_bytes + expected_shared_arena
        );

        let required = qwen_moe_prefill_device_route_required_bytes(
            N_TOKENS,
            N_EXPERT,
            N_EXPERT_USED,
            HIDDEN_DIM,
            FFN_DIM,
        )
        .expect("device-route required bytes");
        assert_eq!(
            qwen_moe_prefill_device_route_preflight(
                true,
                true,
                N_TOKENS,
                N_EXPERT,
                N_EXPERT_USED,
                HIDDEN_DIM,
                FFN_DIM,
                required - 1,
            ),
            Ok(QwenMoeLlamaIdPreflight::Fallback(
                QwenMoeLlamaIdFallbackReason::ScratchOverBudget
            ))
        );
    }

    #[test]
    fn qwen_moe_device_route_q8_scratch_follows_chain_threshold() {
        let short = match qwen_moe_prefill_device_route_plan(
            true,
            true,
            QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD - 1,
            8,
            8,
            256,
            256,
        )
        .expect("short device-route plan")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("unexpected short device-route fallback: {reason:?}")
            }
        };
        let long = match qwen_moe_prefill_device_route_plan(
            true,
            true,
            QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD,
            8,
            8,
            256,
            256,
        )
        .expect("long device-route plan")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("unexpected long device-route fallback: {reason:?}")
            }
        };
        let generic_long = match qwen_moe_llama_id_preflight(
            true,
            true,
            QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD,
            8,
            8,
            256,
            256,
            usize::MAX,
        )
        .expect("generic long plan")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("unexpected generic long fallback: {reason:?}")
            }
        };

        assert_ne!(short.q8_bytes, 0);
        assert_eq!(long.q8_bytes, 0);
        assert_eq!(
            long.total_bytes,
            generic_long.total_bytes - generic_long.q8_bytes
        );
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::compute::build_metal_context;
    use objc2_metal::MTLBlitCommandEncoder;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn qwen_decode_route_shared_oracle(
        logits: &[f32],
        input: &[f32],
        shared_scale: &[f32],
        n_used: usize,
        shared_expert_id: u32,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut best_values = vec![f32::NEG_INFINITY; n_used];
        let mut ranked = vec![0u32; n_used];
        for (expert, &value) in logits.iter().enumerate() {
            for rank in 0..n_used {
                if value > best_values[rank]
                    || (value == best_values[rank] && (expert as u32) < ranked[rank])
                {
                    for shift in (rank + 1..n_used).rev() {
                        best_values[shift] = best_values[shift - 1];
                        ranked[shift] = ranked[shift - 1];
                    }
                    best_values[rank] = value;
                    ranked[rank] = expert as u32;
                    break;
                }
            }
        }

        let selected_max = best_values[0];
        let mut selected_sum = 0.0f32;
        let mut weights = Vec::with_capacity(n_used + 1);
        for &value in &best_values {
            let weight = (value - selected_max).exp();
            selected_sum += weight;
            weights.push(weight);
        }
        for weight in &mut weights {
            *weight = if selected_sum != 0.0 {
                *weight / selected_sum
            } else {
                0.0
            };
        }

        let mut shared_dot = 0.0f32;
        for (&value, &scale) in input.iter().zip(shared_scale) {
            shared_dot += value * scale;
        }
        ranked.push(shared_expert_id);
        weights.push(1.0f32 / (1.0f32 + (-shared_dot).exp()));
        (ranked, weights)
    }

    fn qwen_decode_route_shared_run(
        ctx: &MetalContext,
        logits: &[f32],
        input: &[f32],
        shared_scale: &[f32],
        n_used: usize,
        shared_expert_id: u32,
    ) -> (Vec<u32>, Vec<f32>) {
        const ID_GUARD: u32 = 0xdead_beef;
        const WEIGHT_GUARD: f32 = -12345.25;

        assert!(logits.len() <= 256);
        assert!((1..=31).contains(&n_used));
        assert!(n_used <= logits.len());
        assert_eq!(input.len(), shared_scale.len());

        let logits_buf = shared_f32_buf(ctx, logits);
        let input_buf = shared_f32_buf(ctx, input);
        let shared_scale_buf = shared_f32_buf(ctx, shared_scale);
        let expert_ids_buf = shared_u32_buf(ctx, &vec![ID_GUARD; n_used + 3]);
        let route_weights_buf = shared_f32_buf(ctx, &vec![WEIGHT_GUARD; n_used + 3]);
        let n_expert_buf = u32_buf(ctx, logits.len() as u32);
        let n_used_buf = u32_buf(ctx, n_used as u32);
        let hidden_dim_buf = u32_buf(ctx, input.len() as u32);
        let shared_expert_id_buf = u32_buf(ctx, shared_expert_id);

        let command = ctx.queue.commandBuffer().expect("route command buffer");
        let encoder = command
            .computeCommandEncoder()
            .expect("route compute encoder");
        encode_qwen_moe_decode_route_shared(
            ctx,
            &encoder,
            &logits_buf,
            &input_buf,
            &shared_scale_buf,
            &expert_ids_buf,
            &route_weights_buf,
            &n_expert_buf,
            &n_used_buf,
            &hidden_dim_buf,
            &shared_expert_id_buf,
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let all_ids = unsafe {
            std::slice::from_raw_parts(expert_ids_buf.contents().as_ptr() as *const u32, n_used + 3)
        };
        let all_weights = readback(&route_weights_buf, n_used + 3);
        assert_eq!(&all_ids[n_used + 1..], &[ID_GUARD, ID_GUARD]);
        for &guard in &all_weights[n_used + 1..] {
            assert_eq!(guard.to_bits(), WEIGHT_GUARD.to_bits());
        }
        (
            all_ids[..n_used + 1].to_vec(),
            all_weights[..n_used + 1].to_vec(),
        )
    }

    fn assert_decode_route_weights_close(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        let len = got.len();
        for (rank, (&got, &expected)) in got.iter().zip(expected).enumerate() {
            let tolerance = if rank + 1 == len {
                3.0e-6f32
            } else {
                2.0e-6f32
            };
            assert!(
                (got - expected).abs() <= tolerance,
                "route weight {rank}: got={got:?}, expected={expected:?}, tolerance={tolerance:?}"
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_decode_route_shared_parallel_actual_shape_matches_cpu_oracle() {
        const N_EXPERT: usize = 256;
        const N_USED: usize = 8;
        const HIDDEN_DIM: usize = 2048;
        const SHARED_EXPERT_ID: u32 = 256;

        let ctx = build_metal_context().expect("no metal device");
        let mut logits = (0..N_EXPERT)
            .map(|expert| -20.0f32 - expert as f32 * (1.0f32 / 128.0f32))
            .collect::<Vec<_>>();
        for (expert, value) in [
            (4usize, 9.0f32),
            (201, 9.0),
            (255, 8.0),
            (31, 7.0),
            (128, 6.0),
            (17, 5.0),
            (99, 5.0),
            (140, 5.0),
            (240, 5.0),
        ] {
            logits[expert] = value;
        }
        let input = (0..HIDDEN_DIM)
            .map(|i| ((i % 29) as i32 - 14) as f32 * (1.0f32 / 32.0f32))
            .collect::<Vec<_>>();
        let shared_scale = (0..HIDDEN_DIM)
            .map(|i| (((i * 7) % 31) as i32 - 15) as f32 * (1.0f32 / 64.0f32))
            .collect::<Vec<_>>();

        let expected = qwen_decode_route_shared_oracle(
            &logits,
            &input,
            &shared_scale,
            N_USED,
            SHARED_EXPERT_ID,
        );
        assert_eq!(
            &expected.0[..N_USED],
            &[4, 201, 255, 31, 128, 17, 99, 140],
            "tie ordering and top-8 boundary"
        );
        let got = qwen_decode_route_shared_run(
            &ctx,
            &logits,
            &input,
            &shared_scale,
            N_USED,
            SHARED_EXPERT_ID,
        );
        assert_eq!(got.0, expected.0);
        assert_decode_route_weights_close(&got.1, &expected.1);

        let boundary_logits = vec![1.25f32; 37];
        let boundary_input = (0..513)
            .map(|i| ((i % 11) as i32 - 5) as f32 * (1.0f32 / 16.0f32))
            .collect::<Vec<_>>();
        let boundary_scale = (0..513)
            .map(|i| (((i * 5) % 13) as i32 - 6) as f32 * (1.0f32 / 32.0f32))
            .collect::<Vec<_>>();
        let boundary_expected = qwen_decode_route_shared_oracle(
            &boundary_logits,
            &boundary_input,
            &boundary_scale,
            31,
            37,
        );
        assert_eq!(
            &boundary_expected.0[..31],
            &(0u32..31).collect::<Vec<_>>(),
            "n_used=31 tie boundary"
        );
        let boundary_got = qwen_decode_route_shared_run(
            &ctx,
            &boundary_logits,
            &boundary_input,
            &boundary_scale,
            31,
            37,
        );
        assert_eq!(boundary_got.0, boundary_expected.0);
        assert_decode_route_weights_close(&boundary_got.1, &boundary_expected.1);

        let sparse_logits = [
            f32::NEG_INFINITY,
            f32::NAN,
            3.0,
            f32::NEG_INFINITY,
            1.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            2.0,
            f32::NEG_INFINITY,
        ];
        let sparse_expected =
            qwen_decode_route_shared_oracle(&sparse_logits, &[0.25, -0.5], &[2.0, 1.0], 8, 9);
        assert_eq!(
            sparse_expected.0,
            &[2, 7, 4, 0, 0, 0, 0, 0, 9],
            "non-selectable logits preserve scalar expert IDs"
        );
        let sparse_got =
            qwen_decode_route_shared_run(&ctx, &sparse_logits, &[0.25, -0.5], &[2.0, 1.0], 8, 9);
        assert_eq!(sparse_got.0, sparse_expected.0);
        assert_decode_route_weights_close(&sparse_got.1, &sparse_expected.1);
    }

    #[test]
    fn qwen_moe_device_route_alias_fits_full_prompt_budget() {
        const BUDGET: usize = 512 * 1024 * 1024;
        const REQUIRED: usize = 532_189_184;
        assert_eq!(
            qwen_moe_prefill_device_route_required_bytes(4556, 256, 8, 2048, 512),
            Ok(REQUIRED)
        );
        assert!(matches!(
            qwen_moe_prefill_device_route_preflight(true, true, 4556, 256, 8, 2048, 512, BUDGET,),
            Ok(QwenMoeLlamaIdPreflight::Run(_))
        ));

        let ctx = build_metal_context().expect("no metal device");
        let tiny_plan =
            match qwen_moe_prefill_device_route_preflight(true, true, 2, 4, 2, 256, 256, BUDGET)
                .expect("device-route preflight")
            {
                QwenMoeLlamaIdPreflight::Run(plan) => plan,
                QwenMoeLlamaIdPreflight::Fallback(reason) => {
                    panic!("unexpected device-route fallback: {reason:?}")
                }
            };
        let carrier =
            QwenMoeLlamaIdCarrier::new_device_routes(&ctx, tiny_plan).expect("device carrier");
        assert!(carrier.prefill_route_norm_aliases_sparse_output());
        assert!(!carrier.q8_aliases_gate());
        assert_eq!(
            carrier.allocation_bytes(),
            qwen_moe_prefill_device_route_required_bytes(2, 4, 2, 256, 256)
                .expect("tiny required bytes")
        );

        let long_plan = match qwen_moe_prefill_device_route_preflight(
            true,
            true,
            QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD,
            1,
            1,
            1,
            1,
            BUDGET,
        )
        .expect("long device-route preflight")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("unexpected long device-route fallback: {reason:?}")
            }
        };
        assert_eq!(long_plan.q8_bytes, 0);
        let long_carrier =
            QwenMoeLlamaIdCarrier::new_device_routes(&ctx, long_plan).expect("long carrier");
        assert!(long_carrier.q8_aliases_gate());
        assert_eq!(
            long_carrier.allocation_bytes(),
            qwen_moe_prefill_device_route_required_bytes(
                QWEN_MOE_CHAIN_Q8K_TOKEN_THRESHOLD,
                1,
                1,
                1,
                1,
            )
            .expect("long required bytes")
        );
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_device_route_top1_shared_q4_dispatch_has_backing_arena() {
        let ctx =
            crate::compute::build_metal_context_with_opts(false, true).expect("no metal device");
        if !ctx.tensorops_capable {
            eprintln!("skipping: Metal tensor operations unavailable");
            return;
        }

        const N_TOKENS: usize = 128;
        const N_EXPERT: usize = 4;
        const N_EXPERT_USED: usize = 1;
        const HIDDEN_DIM: usize = 256;
        const FFN_DIM: usize = 1024;
        let plan = match qwen_moe_prefill_device_route_preflight(
            true,
            true,
            N_TOKENS,
            N_EXPERT,
            N_EXPERT_USED,
            HIDDEN_DIM,
            FFN_DIM,
            usize::MAX,
        )
        .expect("device-route top-1 preflight")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("unexpected device-route fallback: {reason:?}")
            }
        };
        let mut carrier =
            QwenMoeLlamaIdCarrier::new_device_routes(&ctx, plan).expect("device-route carrier");
        carrier
            .refresh_routes(
                &vec![0; N_TOKENS],
                &vec![1.0; N_TOKENS],
                &vec![0.5; N_TOKENS],
            )
            .expect("top-1 routes");

        let norm_all_dev = shared_f32_buf(&ctx, &qwen_moe_llama_test_input(N_TOKENS, HIDDEN_DIM));
        let (shared_gate_bytes, _) =
            qwen_moe_llama_test_shared_matrix(QwenMoeLlamaIdQuant::Q4K, FFN_DIM, HIDDEN_DIM, 7001);
        let (shared_up_bytes, _) =
            qwen_moe_llama_test_shared_matrix(QwenMoeLlamaIdQuant::Q4K, FFN_DIM, HIDDEN_DIM, 8009);
        let (shared_down_bytes, _) =
            qwen_moe_llama_test_shared_matrix(QwenMoeLlamaIdQuant::Q6K, HIDDEN_DIM, FFN_DIM, 9001);
        let shared_gate_dev = shared_u8_buf(&ctx, &shared_gate_bytes);
        let shared_up_dev = shared_u8_buf(&ctx, &shared_up_bytes);
        let shared_down_dev = shared_u8_buf(&ctx, &shared_down_bytes);

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = crate::compute::chain_compute_encoder(&ctx, &command);
        encode_qwen_moe_shared_q4_canonical_ops(
            &ctx,
            &encoder,
            &carrier,
            &norm_all_dev,
            &shared_gate_dev,
            0,
            &shared_up_dev,
            0,
            &shared_down_dev,
            0,
            QwenMoeLlamaIdQuant::Q6K,
        )
        .expect("shared-Q4 top-1 encode");
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_chain_hybrid_mixed_15_16_matches_small_id_and_large_generic_bits() {
        let ctx =
            crate::compute::build_metal_context_with_opts(false, true).expect("no metal device");
        if !ctx.tensorops_capable {
            eprintln!("skipping: Metal tensor operations unavailable");
            return;
        }

        const N_TOKENS: usize = 31;
        const N_EXPERT: usize = 2;
        const N_EXPERT_USED: usize = 1;
        const DIM: usize = 256;
        const SMALL_COUNT: usize = 15;
        const LARGE_COUNT: usize = 16;
        let selected_experts = (0..N_TOKENS)
            .map(|token| u32::from(token >= SMALL_COUNT))
            .collect::<Vec<_>>();
        let plan = match qwen_moe_prefill_device_route_preflight(
            true,
            true,
            N_TOKENS,
            N_EXPERT,
            N_EXPERT_USED,
            DIM,
            DIM,
            usize::MAX,
        )
        .expect("mixed route preflight")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("unexpected mixed route fallback: {reason:?}")
            }
        };
        let mut carrier =
            QwenMoeLlamaIdCarrier::new_device_routes(&ctx, plan).expect("mixed route carrier");
        carrier
            .refresh_routes(
                &selected_experts,
                &vec![1.0; N_TOKENS],
                &vec![0.5; N_TOKENS],
            )
            .expect("mixed routes");
        let input = qwen_moe_llama_test_input(N_TOKENS, DIM);
        let input_f32_dev = shared_f32_buf(&ctx, &input);
        let input_f16_dev = empty_f16_buf(&ctx, N_TOKENS * DIM);
        let large_f16_dev = empty_f16_buf(&ctx, LARGE_COUNT * DIM);
        let input_f16 = input
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        unsafe {
            std::slice::from_raw_parts_mut(
                input_f16_dev.contents().as_ptr() as *mut u16,
                input_f16.len(),
            )
            .copy_from_slice(&input_f16);
            std::slice::from_raw_parts_mut(
                large_f16_dev.contents().as_ptr() as *mut u16,
                LARGE_COUNT * DIM,
            )
            .copy_from_slice(&input_f16[SMALL_COUNT * DIM..]);
        }

        for quant in [QwenMoeLlamaIdQuant::Q4K, QwenMoeLlamaIdQuant::Q6K] {
            let (weight_bytes, _) =
                qwen_moe_llama_test_sparse_matrix(quant, N_EXPERT, DIM, DIM, 6101);
            let weights = shared_u8_buf(&ctx, &weight_bytes);
            let expert_weight_bytes = weight_bytes.len() / N_EXPERT;
            let shape = QwenMoeLlamaIdMatmulShape {
                input_dim: DIM,
                output_dim: DIM,
                n_tokens: N_TOKENS,
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                expert_weight_bytes,
            };
            let reference_out = empty_f32_buf(&ctx, N_TOKENS * DIM);
            let hybrid_out = empty_f32_buf(&ctx, N_TOKENS * DIM);
            let generic_large_out = empty_f32_buf(&ctx, LARGE_COUNT * DIM);

            let command = ctx.queue.commandBuffer().expect("command buffer");
            let encoder = crate::compute::chain_compute_encoder(&ctx, &command);
            qwen_moe_llama_id_map0_encode(
                &ctx,
                &encoder,
                &carrier.selected_experts_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                N_TOKENS,
                N_EXPERT_USED,
                N_EXPERT,
            )
            .expect("mixed map");
            crate::compute::chain_barrier(&ctx, &encoder);
            qwen_moe_llama_id_build_blocks_encode(
                &ctx,
                &encoder,
                &carrier.tpe_dev,
                &carrier.block_meta_dev,
                carrier.n_tokens,
                carrier.n_expert_used,
                carrier.n_expert,
                carrier.hidden_dim,
                carrier.ffn_dim,
            )
            .expect("mixed compact blocks");
            crate::compute::chain_barrier(&ctx, &encoder);
            qwen_moe_llama_quantize_q8k_f32_encode(
                &ctx,
                &encoder,
                &input_f32_dev,
                &carrier.q8_dev,
                N_TOKENS * DIM,
            )
            .expect("mixed q8k input");
            crate::compute::chain_barrier(&ctx, &encoder);
            qwen_moe_llama_mul_mm_id_q8k_encode(
                &ctx,
                &encoder,
                quant,
                &weights,
                0,
                &carrier.q8_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                &reference_out,
                shape,
            )
            .expect("small Q8_K ID reference");
            encode_qwen_moe_chain_hybrid_mul_mm_id(
                &ctx,
                &encoder,
                quant,
                &weights,
                0,
                &carrier.q8_dev,
                &input_f16_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                &carrier.block_meta_dev,
                carrier.block_local_offset,
                carrier.block_indirect_offset
                    + match quant {
                        QwenMoeLlamaIdQuant::Q4K => QWEN_MOE_LLAMA_ID_FFN_Q4_INDIRECT_OFFSET,
                        QwenMoeLlamaIdQuant::Q6K => QWEN_MOE_LLAMA_ID_HIDDEN_Q6_INDIRECT_OFFSET,
                        _ => unreachable!(),
                    },
                true,
                &hybrid_out,
                shape,
                N_TOKENS,
            )
            .expect("mixed hybrid");
            match quant {
                QwenMoeLlamaIdQuant::Q4K => {
                    crate::compute::encode_gemm_q4k_tensorops_v2_offset(
                        &ctx,
                        &encoder,
                        &weights,
                        expert_weight_bytes,
                        &large_f16_dev,
                        0,
                        &generic_large_out,
                        0,
                        DIM,
                        DIM,
                        LARGE_COUNT,
                    )
                    .expect("generic Q4 large reference");
                }
                QwenMoeLlamaIdQuant::Q6K => {
                    crate::compute::encode_gemm_q6k_tensorops_v2_offset(
                        &ctx,
                        &encoder,
                        &weights,
                        expert_weight_bytes,
                        &large_f16_dev,
                        0,
                        &generic_large_out,
                        0,
                        DIM,
                        DIM,
                        LARGE_COUNT,
                    )
                    .expect("generic Q6 large reference");
                }
                _ => unreachable!(),
            }
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

            let reference = readback(&reference_out, N_TOKENS * DIM);
            let hybrid = readback(&hybrid_out, N_TOKENS * DIM);
            let generic_large = readback(&generic_large_out, LARGE_COUNT * DIM);
            for slot in 0..SMALL_COUNT {
                for row in 0..DIM {
                    assert_eq!(
                        hybrid[slot * DIM + row].to_bits(),
                        reference[slot * DIM + row].to_bits(),
                        "{quant:?} small slot={slot} row={row}"
                    );
                }
            }
            for local in 0..LARGE_COUNT {
                let slot = SMALL_COUNT + local;
                for row in 0..DIM {
                    assert_eq!(
                        hybrid[slot * DIM + row].to_bits(),
                        generic_large[local * DIM + row].to_bits(),
                        "{quant:?} large slot={slot} row={row}"
                    );
                }
            }
        }
    }

    struct QwenRouteParityOutput {
        normed: Vec<f32>,
        logits: Vec<f32>,
        expert_ids: Vec<u32>,
        route_weights: Vec<f32>,
        shared_weights: Vec<f32>,
    }

    fn qwen_route_parity_inputs(
        n_tokens: usize,
        hidden_dim: usize,
        n_expert: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let magnitude = f32::from_bits(0x3f7f_fff8);
        let hidden = (0..n_tokens * hidden_dim)
            .map(|i| {
                if (i.wrapping_mul(13).wrapping_add(i / hidden_dim * 17)) % 7 < 3 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect::<Vec<_>>();
        let norm_weight = (0..hidden_dim)
            .map(|col| [0.25f32, 0.5, 1.0, 2.0][col % 4])
            .collect::<Vec<_>>();
        let mut router_weight = vec![0.0f32; n_expert * hidden_dim];
        for expert in 0..n_expert {
            for tap in 0..4 {
                let col = (expert * 37 + tap * 499 + 11) % hidden_dim;
                let coefficient = ((expert * 19 + tap * 7 + 3) % 17) as i32 - 8;
                router_weight[expert * hidden_dim + col] = coefficient as f32 * (1.0f32 / 256.0f32);
            }
        }
        if n_expert >= 2 {
            let (first, rest) = router_weight.split_at_mut(hidden_dim);
            rest[..hidden_dim].copy_from_slice(first);
        }
        if n_expert >= 2 {
            let nan_row = n_expert - 2;
            let neg_inf_row = n_expert - 1;
            router_weight[nan_row * hidden_dim] = f32::NAN;
            router_weight[neg_inf_row * hidden_dim] = f32::NEG_INFINITY;
        }
        let mut shared_scale = vec![0.0f32; hidden_dim];
        for tap in 0..16 {
            let col = (tap * 127 + 5) % hidden_dim;
            shared_scale[col] =
                ((tap * 5 + 3) % 11) as f32 * (1.0f32 / 512.0f32) - (5.0f32 / 512.0f32);
        }
        (hidden, norm_weight, router_weight, shared_scale)
    }

    fn qwen_route_cpu_f32_dot(weight: &[f32], input: &[f32]) -> f32 {
        assert_eq!(weight.len(), input.len());
        let mut dot = 0.0f32;
        for col in 0..input.len() {
            let product = weight[col] * input[col];
            dot += product;
        }
        dot
    }

    fn qwen_route_cpu_router_dot(weight: &[f32], input: &[f32]) -> f32 {
        assert_eq!(weight.len(), input.len());
        let mut acc0 = [0.0f32; 4];
        let mut acc1 = [0.0f32; 4];
        let mut col = 0;
        while col + 8 <= input.len() {
            for lane in 0..4 {
                acc0[lane] = weight[col + lane].mul_add(input[col + lane], acc0[lane]);
                acc1[lane] = weight[col + 4 + lane].mul_add(input[col + 4 + lane], acc1[lane]);
            }
            col += 8;
        }
        if col + 4 <= input.len() {
            for lane in 0..4 {
                acc0[lane] = weight[col + lane].mul_add(input[col + lane], acc0[lane]);
            }
            col += 4;
        }
        let combined = [
            acc0[0] + acc1[0],
            acc0[1] + acc1[1],
            acc0[2] + acc1[2],
            acc0[3] + acc1[3],
        ];
        let mut dot = (combined[0] + combined[1]) + (combined[2] + combined[3]);
        while col < input.len() {
            let product = weight[col] * input[col];
            dot += product;
            col += 1;
        }
        dot
    }

    fn qwen_route_cpu_topk(
        logits: &[f32],
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut expert_ids = vec![0u32; n_tokens * n_expert_used];
        let mut route_weights = vec![0.0f32; n_tokens * n_expert_used];
        for token in 0..n_tokens {
            let mut best_values = [f32::NEG_INFINITY; 32];
            let mut best_ids = [usize::MAX; 32];
            let row = &logits[token * n_expert..(token + 1) * n_expert];
            for (expert, &value) in row.iter().enumerate() {
                for rank in 0..n_expert_used {
                    if value > best_values[rank]
                        || (value == best_values[rank] && expert < best_ids[rank])
                    {
                        for shift in (rank + 1..n_expert_used).rev() {
                            best_values[shift] = best_values[shift - 1];
                            best_ids[shift] = best_ids[shift - 1];
                        }
                        best_values[rank] = value;
                        best_ids[rank] = expert;
                        break;
                    }
                }
            }
            let selected_max = best_values[0];
            let mut selected_sum = 0.0f32;
            let base = token * n_expert_used;
            for rank in 0..n_expert_used {
                let weight = (best_values[rank] - selected_max).exp();
                expert_ids[base + rank] = best_ids[rank] as u32;
                route_weights[base + rank] = weight;
                selected_sum += weight;
            }
            if selected_sum != 0.0 {
                for weight in &mut route_weights[base..base + n_expert_used] {
                    *weight /= selected_sum;
                }
            }
        }
        (expert_ids, route_weights)
    }

    fn qwen_route_parity_oracle(
        hidden: &[f32],
        norm_weight: &[f32],
        router_weight: &[f32],
        shared_scale: &[f32],
        norm_eps: f32,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
    ) -> QwenRouteParityOutput {
        let hidden_dim = norm_weight.len();
        let mut normed = vec![0.0f32; n_tokens * hidden_dim];
        for token in 0..n_tokens {
            let base = token * hidden_dim;
            let mean_square = hidden[base..base + hidden_dim]
                .iter()
                .map(|&value| (value as f64) * (value as f64))
                .sum::<f64>()
                / hidden_dim as f64;
            let rms = (mean_square + norm_eps as f64).sqrt() as f32;
            for col in 0..hidden_dim {
                normed[base + col] = (hidden[base + col] / rms) * norm_weight[col];
            }
        }

        let mut logits = vec![0.0f32; n_tokens * n_expert];
        let mut shared_dots = vec![0.0f32; n_tokens];
        for token in 0..n_tokens {
            let input = &normed[token * hidden_dim..(token + 1) * hidden_dim];
            for expert in 0..n_expert {
                let weight = &router_weight[expert * hidden_dim..(expert + 1) * hidden_dim];
                let dot = qwen_route_cpu_router_dot(weight, input);
                logits[token * n_expert + expert] = dot;
            }
            shared_dots[token] = qwen_route_cpu_f32_dot(shared_scale, input);
        }

        let (expert_ids, route_weights) =
            qwen_route_cpu_topk(&logits, n_tokens, n_expert, n_expert_used);
        let shared_weights = shared_dots
            .into_iter()
            .map(|dot| 1.0f32 / (1.0f32 + (-dot).exp()))
            .collect();
        QwenRouteParityOutput {
            normed,
            logits,
            expert_ids,
            route_weights,
            shared_weights,
        }
    }

    fn qwen_route_parity_run(
        ctx: &MetalContext,
        hidden: &[f32],
        norm_weight: &[f32],
        router_weight: &[f32],
        shared_scale: &[f32],
        norm_eps: f32,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
    ) -> QwenRouteParityOutput {
        let hidden_dim = norm_weight.len();
        let plan = match qwen_moe_prefill_device_route_preflight(
            true,
            true,
            n_tokens,
            n_expert,
            n_expert_used,
            hidden_dim,
            512,
            usize::MAX,
        )
        .expect("route parity preflight")
        {
            QwenMoeLlamaIdPreflight::Run(plan) => plan,
            QwenMoeLlamaIdPreflight::Fallback(reason) => {
                panic!("route parity unexpectedly fell back: {reason:?}")
            }
        };
        let carrier =
            QwenMoeLlamaIdCarrier::new_device_routes(ctx, plan).expect("route parity carrier");
        assert!(carrier.prefill_route_norm_aliases_sparse_output());
        let route = carrier.prefill_route.as_ref().expect("route scratch");
        let hidden_buffer = shared_f32_buf(ctx, hidden);
        let norm_buffer = shared_f32_buf(ctx, norm_weight);
        let router_buffer = shared_f32_buf(ctx, router_weight);
        let shared_scale_buffer = shared_f32_buf(ctx, shared_scale);
        let norm_stage = empty_f32_buf(ctx, n_tokens * hidden_dim);
        let logits_stage = empty_f32_buf(ctx, n_tokens * n_expert);
        let expert_stage = shared_u32_buf(ctx, &vec![0; n_tokens * n_expert_used]);
        let route_stage = empty_f32_buf(ctx, n_tokens * n_expert_used);
        let shared_stage = empty_f32_buf(ctx, n_tokens);

        let command = ctx.queue.commandBuffer().expect("route parity command");
        let encoder = crate::compute::chain_compute_encoder(ctx, &command);
        encode_qwen_prefill_route_ops(
            ctx,
            &encoder,
            &hidden_buffer,
            &norm_buffer,
            &router_buffer,
            &shared_scale_buffer,
            norm_eps,
            &carrier,
            QwenRouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1,
        )
        .expect("encode route parity");
        encoder.endEncoding();
        let blit = command.blitCommandEncoder().expect("route parity blit");
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &route.normed_dev,
                0,
                &norm_stage,
                0,
                n_tokens * hidden_dim * 4,
            );
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &route.router_logits_dev,
                0,
                &logits_stage,
                0,
                n_tokens * n_expert * 4,
            );
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &carrier.selected_experts_dev,
                0,
                &expert_stage,
                0,
                n_tokens * n_expert_used * 4,
            );
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &carrier.route_weights_dev,
                0,
                &route_stage,
                0,
                n_tokens * n_expert_used * 4,
            );
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &carrier.shared_route_weights_dev,
                0,
                &shared_stage,
                0,
                n_tokens * 4,
            );
        }
        blit.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let expert_ids = unsafe {
            std::slice::from_raw_parts(
                expert_stage.contents().as_ptr() as *const u32,
                n_tokens * n_expert_used,
            )
            .to_vec()
        };
        QwenRouteParityOutput {
            normed: readback(&norm_stage, n_tokens * hidden_dim),
            logits: readback(&logits_stage, n_tokens * n_expert),
            expert_ids,
            route_weights: readback(&route_stage, n_tokens * n_expert_used),
            shared_weights: readback(&shared_stage, n_tokens),
        }
    }

    fn assert_f32_bits(stage: &str, got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len(), "{stage} length");
        for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{stage} first bit mismatch at {index}: got={got:?} ({:#010x}) expected={expected:?} ({:#010x})",
                got.to_bits(),
                expected.to_bits(),
            );
        }
    }

    fn qwen_route_parity_assert_case(
        ctx: &MetalContext,
        n_tokens: usize,
        hidden_dim: usize,
        n_expert: usize,
        n_expert_used: usize,
    ) {
        let norm_eps = 3.0e-5f32;
        let (hidden, norm_weight, router_weight, shared_scale) =
            qwen_route_parity_inputs(n_tokens, hidden_dim, n_expert);
        let expected = qwen_route_parity_oracle(
            &hidden,
            &norm_weight,
            &router_weight,
            &shared_scale,
            norm_eps,
            n_tokens,
            n_expert,
            n_expert_used,
        );
        let got = qwen_route_parity_run(
            ctx,
            &hidden,
            &norm_weight,
            &router_weight,
            &shared_scale,
            norm_eps,
            n_tokens,
            n_expert,
            n_expert_used,
        );
        assert_f32_bits("norm", &got.normed, &expected.normed);
        assert_f32_bits("router logits", &got.logits, &expected.logits);
        assert_eq!(got.expert_ids, expected.expert_ids, "selected expert IDs");
        assert_f32_bits(
            "selected softmax weights",
            &got.route_weights,
            &expected.route_weights,
        );
        assert_f32_bits(
            "shared sigmoid weights",
            &got.shared_weights,
            &expected.shared_weights,
        );
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_prefill_layer_chain_route_actual_shape_matches_cpu_bits() {
        const N_TOKENS: usize = 128;
        const HIDDEN_DIM: usize = 2048;
        const N_EXPERT: usize = 256;
        const N_EXPERT_USED: usize = 8;
        const NORM_EPS: f32 = 2.5e-5;

        let ctx = build_metal_context().expect("no metal device");
        let (mut hidden, norm_weight, router_weight, shared_scale) =
            qwen_route_parity_inputs(N_TOKENS, HIDDEN_DIM, N_EXPERT);
        for token in 0..N_TOKENS {
            for col in 0..HIDDEN_DIM {
                let centered = ((token * 97 + col * 53 + token * col * 3) % 2003) as f32 - 1001.0;
                hidden[token * HIDDEN_DIM + col] = centered * (1.0f32 / 1024.0f32)
                    + ((col % 7) as f32 - 3.0) * (1.0f32 / 65536.0f32);
            }
        }
        let expected = qwen_route_parity_oracle(
            &hidden,
            &norm_weight,
            &router_weight,
            &shared_scale,
            NORM_EPS,
            N_TOKENS,
            N_EXPERT,
            N_EXPERT_USED,
        );
        let got = qwen_route_parity_run(
            &ctx,
            &hidden,
            &norm_weight,
            &router_weight,
            &shared_scale,
            NORM_EPS,
            N_TOKENS,
            N_EXPERT,
            N_EXPERT_USED,
        );
        assert_f32_bits("norm", &got.normed, &expected.normed);
        assert_f32_bits("router logits", &got.logits, &expected.logits);
        assert_eq!(got.expert_ids, expected.expert_ids, "selected expert IDs");
        assert_f32_bits(
            "selected softmax weights",
            &got.route_weights,
            &expected.route_weights,
        );
        assert_f32_bits(
            "shared sigmoid weights",
            &got.shared_weights,
            &expected.shared_weights,
        );
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_prefill_layer_chain_route_rank_boundaries_match_cpu_bits() {
        let ctx = build_metal_context().expect("no metal device");
        qwen_route_parity_assert_case(&ctx, 2, 256, 256, 1);
        qwen_route_parity_assert_case(&ctx, 2, 256, 256, 32);
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_prefill_layer_chain_zero_rank_fails_before_allocation() {
        assert_eq!(
            qwen_moe_prefill_device_route_required_bytes(128, 256, 0, 2048, 512),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        assert_eq!(
            qwen_moe_prefill_device_route_preflight(true, true, 128, 256, 0, 2048, 512, usize::MAX,),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
    }

    struct EnvGuard {
        key: &'static str,
        prev: Option<OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn half_bytes_and_cpu(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
        let mut cpu = Vec::with_capacity(values.len());
        for &value in values {
            let half = half::f16::from_f32(value);
            bytes.extend_from_slice(&half.to_le_bytes());
            cpu.push(half.to_f32());
        }
        (bytes, cpu)
    }

    fn q4_rows_pattern(total_rows: usize, k: usize) -> Vec<u8> {
        use rnb_cpu::quantize::quant::quantize_q4_k_vec;

        assert_eq!(k % 256, 0, "Q4_K K must be a multiple of 256");
        let mut out = Vec::with_capacity(total_rows * (k / 256) * 144);
        for row in 0..total_rows {
            let vals = (0..k)
                .map(|col| ((row * 17 + col * 11 + 5) % 113) as f32 * 0.004 - 0.22)
                .collect::<Vec<_>>();
            out.extend_from_slice(&quantize_q4_k_vec(&vals));
        }
        out
    }

    fn qwen_moe_v4_rank_scatter_reference(
        weight: &[u8],
        n: usize,
        k: usize,
        seq_len: usize,
        n_expert: usize,
        n_expert_used: usize,
        block_bytes: usize,
        plan: &QwenMoeMulMmIdV4Plan,
        act: &[f32],
        dequant_block: fn(&[u8]) -> [f32; 256],
    ) -> Vec<f32> {
        assert_eq!(k % 256, 0);
        assert_eq!(act.len(), plan.token_ids_sorted.len() * k);
        let nb = k / 256;
        let expert_stride = n * nb * block_bytes;
        assert_eq!(weight.len(), n_expert * expert_stride);
        let mut out = vec![0.0f32; seq_len * n];
        let mut row_deq = vec![0.0f32; k];
        for rank in 0..n_expert_used {
            for expert in 0..n_expert {
                let pair = expert * n_expert_used + rank;
                let slot_base = plan.expert_rank_offsets[pair] as usize;
                let count = plan.expert_rank_counts[pair] as usize;
                for local in 0..count {
                    let slot = slot_base + local;
                    let token = plan.token_ids_sorted[slot] as usize;
                    let route = plan.route_weights_sorted[slot];
                    let input = &act[slot * k..(slot + 1) * k];
                    for row in 0..n {
                        let row_base = expert * expert_stride + row * nb * block_bytes;
                        for sb in 0..nb {
                            let block = &weight
                                [row_base + sb * block_bytes..row_base + (sb + 1) * block_bytes];
                            row_deq[sb * 256..(sb + 1) * 256]
                                .copy_from_slice(&dequant_block(block));
                        }
                        let dot = row_deq
                            .iter()
                            .zip(input.iter())
                            .map(|(&w, &x)| w * x)
                            .sum::<f32>();
                        out[token * n + row] += route * dot;
                    }
                }
            }
        }
        out
    }

    fn assert_global_rel_close(name: &str, got: &[f32], want: &[f32], limit: f32) {
        assert_eq!(got.len(), want.len(), "{name} length");
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let mut max_abs = 0.0f32;
        for (&g, &w) in got.iter().zip(want.iter()) {
            let diff = g - w;
            num += (diff as f64) * (diff as f64);
            den += (w as f64) * (w as f64);
            max_abs = max_abs.max(diff.abs());
        }
        let global_rel = (num.sqrt() / den.sqrt().max(1e-9)) as f32;
        assert!(
            global_rel <= limit,
            "{name} global_rel={global_rel:.3e} max_abs={max_abs:.3e} limit={limit:.3e}"
        );
    }

    fn cpu_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let dim = weight.len();
        let mean_sq: f32 = input.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let rms = (mean_sq + eps).sqrt();
        (0..dim).map(|i| (input[i] / rms) * weight[i]).collect()
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn rms_norm_matches_cpu() {
        let ctx = build_metal_context().expect("no metal device");
        let dim = 3072usize; // llama-3.2-3B hidden_dim (256 배수 아님 — grid-stride 검증)
        let input: Vec<f32> = (0..dim).map(|i| ((i % 17) as f32 - 8.0) * 0.13).collect();
        let weight: Vec<f32> = (0..dim).map(|i| 1.0 + ((i % 7) as f32) * 0.01).collect();
        let eps = 1e-5f32;

        let in_buf = shared_f32_buf(&ctx, &input);
        let w_buf = shared_f32_buf(&ctx, &weight);
        let out_buf = empty_f32_buf(&ctx, dim);
        let dim_buf = u32_buf(&ctx, dim as u32);
        let eps_buf = f32_buf(&ctx, eps);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_rms_norm(&ctx, &enc, &in_buf, &w_buf, &out_buf, &dim_buf, &eps_buf);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let gpu = readback(&out_buf, dim);
        let cpu = cpu_rms_norm(&input, &weight, eps);
        for i in 0..dim {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-6);
            assert!(rel < 1e-3, "i={i} gpu={} cpu={} rel={rel}", gpu[i], cpu[i]);
        }
    }

    fn cpu_silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
        (0..gate.len())
            .map(|i| (gate[i] / (1.0 + (-gate[i]).exp())) * up[i])
            .collect()
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn silu_mul_matches_cpu() {
        let ctx = build_metal_context().expect("no metal device");
        let dim = 8192usize; // llama-3.2-3B ffn_dim
        let gate: Vec<f32> = (0..dim).map(|i| ((i % 23) as f32 - 11.0) * 0.2).collect();
        let up: Vec<f32> = (0..dim).map(|i| ((i % 13) as f32 - 6.0) * 0.15).collect();

        let gate_buf = shared_f32_buf(&ctx, &gate);
        let up_buf = shared_f32_buf(&ctx, &up);
        let dim_buf = u32_buf(&ctx, dim as u32);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_silu_mul(&ctx, &enc, &gate_buf, &up_buf, &dim_buf, dim);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let gpu = readback(&gate_buf, dim);
        let cpu = cpu_silu_mul(&gate, &up);
        for i in 0..dim {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-6);
            assert!(rel < 1e-3, "i={i} gpu={} cpu={} rel={rel}", gpu[i], cpu[i]);
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn silu_mul_to_f16_matches_silu_then_cast_bits() {
        let ctx = build_metal_context().expect("no metal device");
        let dim = 8192usize + 17;
        let gate: Vec<f32> = (0..dim).map(|i| ((i % 29) as f32 - 14.0) * 0.17).collect();
        let up: Vec<f32> = (0..dim).map(|i| ((i % 19) as f32 - 9.0) * 0.11).collect();

        let legacy_gate_buf = shared_f32_buf(&ctx, &gate);
        let legacy_up_buf = shared_f32_buf(&ctx, &up);
        let legacy_half_buf = empty_f16_buf(&ctx, dim);
        let fused_gate_buf = shared_f32_buf(&ctx, &gate);
        let fused_up_buf = shared_f32_buf(&ctx, &up);
        let fused_half_buf = empty_f16_buf(&ctx, dim);
        let dim_buf = u32_buf(&ctx, dim as u32);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_silu_mul(&ctx, &enc, &legacy_gate_buf, &legacy_up_buf, &dim_buf, dim);
        crate::compute::encode_cast_f32_to_f16(
            &ctx,
            &enc,
            &legacy_gate_buf,
            &legacy_half_buf,
            &dim_buf,
            dim,
        );
        crate::compute::encode_silu_mul_to_f16(
            &ctx,
            &enc,
            &fused_gate_buf,
            &fused_up_buf,
            &fused_half_buf,
            &dim_buf,
            dim,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let legacy = unsafe {
            std::slice::from_raw_parts(legacy_half_buf.contents().as_ptr() as *const u16, dim)
        };
        let fused = unsafe {
            std::slice::from_raw_parts(fused_half_buf.contents().as_ptr() as *const u16, dim)
        };
        assert_eq!(legacy, fused);
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn gemv_q4k_device_inout_matches_host() {
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");
        // N=4 rows, K=256 (1 block per row), 같은 block 반복.
        let n = 4usize;
        let k = 256usize;
        let single = tests_fixture::q4k_block_fixed();
        let weight: Vec<u8> = single.iter().cycle().take(n * 144).copied().collect();
        let input: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();

        // host 경로(reference)
        let host = crate::compute::gemv_q4k_with_ctx(&ctx, &weight, &input, n, k);

        // device-in/out 경로: 모든 buffer 를 미리 만들고 encode 만.
        let w_buf = shared_u8_buf(&ctx, &weight);
        let in_buf = shared_f32_buf(&ctx, &input);
        let out_buf = empty_f32_buf(&ctx, n);
        let n_buf = u32_buf(&ctx, n as u32);
        let k_buf = u32_buf(&ctx, k as u32);
        let off_buf = u32_buf(&ctx, 0);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        crate::compute::encode_gemv_q4k(
            &ctx, &enc, &w_buf, &in_buf, &out_buf, &n_buf, &k_buf, &off_buf, n,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let dev = readback(&out_buf, n);
        for i in 0..n {
            let rel = (dev[i] - host[i]).abs() / host[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} dev={} host={} rel={rel}",
                dev[i],
                host[i]
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_q4k_full_arena_nocopy_matches_selected_slice_mmap() {
        use memmap2::MmapOptions;
        use rnb_cpu::quantize::quantize_q4_k_vec;
        use std::fs::OpenOptions;
        use std::io::Write;

        let ctx = build_metal_context().expect("no metal device");
        let n_experts = 40usize;
        let expert = 6usize;
        let n = 512usize;
        let k = 2048usize;
        let slots = 1usize;
        let per_expert_bytes = n * (k / 256) * 144;
        let sparse_bytes_len = n_experts * per_expert_bytes;
        let prefix_len = 15_808usize;

        let input: Vec<f32> = (0..k)
            .map(|i| 0.25 + ((i * 17 + 5) % 43) as f32 * 0.0078125)
            .collect();
        let expert_matrix: Vec<f32> = (0..n * k)
            .map(|i| {
                let row = i / k;
                let col = i % k;
                0.125 + ((row * 13 + col * 7 + 3) % 97) as f32 * 0.00390625
            })
            .collect();
        let expert_quant = quantize_q4_k_vec(&expert_matrix);
        assert_eq!(expert_quant.len(), per_expert_bytes);
        let host = crate::compute::gemv_q4k_with_ctx(&ctx, &expert_quant, &input, n, k);
        let host_norm: f32 = host.iter().map(|v| v.abs()).sum();
        assert!(host_norm > 1.0, "host q4k output unexpectedly near zero");

        let mut sparse_bytes = vec![0u8; sparse_bytes_len];
        let expert_start = expert * per_expert_bytes;
        sparse_bytes[expert_start..expert_start + per_expert_bytes].copy_from_slice(&expert_quant);

        let path = std::env::temp_dir().join(format!(
            "rnb-metal-q4k-arena-{}-{}.bin",
            std::process::id(),
            expert_start
        ));
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create mmap fixture");
            file.write_all(&vec![0xA5u8; prefix_len])
                .expect("write mmap prefix");
            file.write_all(&sparse_bytes).expect("write sparse arena");
            file.sync_all().expect("sync mmap fixture");
        }
        let file = OpenOptions::new()
            .read(true)
            .open(&path)
            .expect("open mmap fixture");
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .expect("map read-only fixture")
        };

        let sparse_raw = &mmap[prefix_len..prefix_len + sparse_bytes_len];
        let selected_raw = &sparse_raw[expert_start..expert_start + per_expert_bytes];
        let (sparse_buf, sparse_page_off) = crate::compute::wrap_nocopy(&ctx, sparse_raw);
        let (selected_buf, selected_page_off) = crate::compute::wrap_nocopy(&ctx, selected_raw);

        let input_buf = shared_f32_buf(&ctx, &input);
        let full_out = empty_f32_buf(&ctx, slots * n);
        let selected_out = empty_f32_buf(&ctx, slots * n);
        let expert_ids = shared_u32_buf(&ctx, &[expert as u32]);
        let n_buf = u32_buf(&ctx, n as u32);
        let k_buf = u32_buf(&ctx, k as u32);
        let per_expert_buf = u64_buf(&ctx, per_expert_bytes as u64);
        let shared_expert_id_buf = u32_buf(&ctx, u32::MAX);
        let sparse_off_buf = u64_buf(&ctx, sparse_page_off as u64);
        let shared_off_buf = u64_buf(&ctx, 0);
        let slots_buf = u32_buf(&ctx, slots as u32);
        let selected_weights = vec![selected_buf];
        let selected_offsets = vec![selected_page_off];

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_qwen_moe_decode_q4k_slots(
            &ctx,
            &enc,
            &sparse_buf,
            &sparse_buf,
            &input_buf,
            &full_out,
            &expert_ids,
            &n_buf,
            &k_buf,
            &per_expert_buf,
            &shared_expert_id_buf,
            &sparse_off_buf,
            &shared_off_buf,
            n,
            slots,
        );
        encode_qwen_moe_decode_q4k_selected_slots(
            &ctx,
            &enc,
            &selected_weights,
            &selected_offsets,
            &input_buf,
            &selected_out,
            &n_buf,
            &k_buf,
            &slots_buf,
            n,
            slots,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let full = readback(&full_out, slots * n);
        let selected = readback(&selected_out, slots * n);
        let _ = std::fs::remove_file(&path);

        let selected_norm: f32 = selected.iter().map(|v| v.abs()).sum();
        assert!(
            selected_norm > 1.0,
            "selected output unexpectedly near zero; host_norm={host_norm} host_head={:?}",
            &host[..4]
        );
        for i in 0..slots * n {
            let rel = (selected[i] - host[i]).abs() / host[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "selected mismatch at {i}: selected={} host={} rel={rel}",
                selected[i],
                host[i]
            );
        }
        for i in 0..slots * n {
            assert!(
                full[i].is_finite(),
                "full arena output became non-finite at {i}: {} selected={}",
                full[i],
                selected[i]
            );
            let rel = (full[i] - selected[i]).abs() / selected[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} full={} selected={} rel={rel} sparse_page_off={} selected_page_off={} expert_start={}",
                full[i],
                selected[i],
                sparse_page_off,
                selected_page_off,
                expert_start
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_q6k_full_arena_nocopy_matches_selected_slice_mmap() {
        use half::f16;
        use memmap2::MmapOptions;
        use std::fs::OpenOptions;
        use std::io::Write;

        let ctx = build_metal_context().expect("no metal device");
        let n_experts = 40usize;
        let expert = 6usize;
        let n = 512usize;
        let k = 2048usize;
        let slots = 1usize;
        let blocks_per_row = k / 256;
        let per_expert_bytes = n * blocks_per_row * 210;
        let sparse_bytes_len = n_experts * per_expert_bytes;
        let prefix_len = 15_808usize;

        let input: Vec<f32> = (0..k)
            .map(|i| ((i * 29 + 17) % 127) as f32 * 0.00390625 - 0.25)
            .collect();
        let mut expert_quant = vec![0u8; per_expert_bytes];
        for row in 0..n {
            for block in 0..blocks_per_row {
                let base = (row * blocks_per_row + block) * 210;
                for i in 0..128 {
                    expert_quant[base + i] = ((row * 13 + block * 7 + i * 3) & 0xff) as u8;
                }
                for i in 0..64 {
                    expert_quant[base + 128 + i] = ((row * 5 + block * 11 + i * 17) & 0xff) as u8;
                }
                for i in 0..16 {
                    expert_quant[base + 192 + i] = (((row + block + i) % 9) as i8 - 4) as u8;
                }
                let d = f16::from_f32(0.0125 + ((row + block) % 7) as f32 * 0.001);
                expert_quant[base + 208..base + 210].copy_from_slice(&d.to_le_bytes());
            }
        }

        let mut sparse_bytes = vec![0u8; sparse_bytes_len];
        let expert_start = expert * per_expert_bytes;
        sparse_bytes[expert_start..expert_start + per_expert_bytes].copy_from_slice(&expert_quant);

        let path = std::env::temp_dir().join(format!(
            "rnb-metal-q6k-arena-{}-{}.bin",
            std::process::id(),
            expert_start
        ));
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create mmap fixture");
            file.write_all(&vec![0xA5u8; prefix_len])
                .expect("write mmap prefix");
            file.write_all(&sparse_bytes).expect("write sparse arena");
            file.sync_all().expect("sync mmap fixture");
        }
        let file = OpenOptions::new()
            .read(true)
            .open(&path)
            .expect("open mmap fixture");
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .expect("map read-only fixture")
        };

        let sparse_raw = &mmap[prefix_len..prefix_len + sparse_bytes_len];
        let selected_raw = &sparse_raw[expert_start..expert_start + per_expert_bytes];
        let (sparse_buf, sparse_page_off) = crate::compute::wrap_nocopy(&ctx, sparse_raw);
        let (selected_buf, selected_page_off) = crate::compute::wrap_nocopy(&ctx, selected_raw);

        let input_buf = shared_f32_buf(&ctx, &input);
        let full_out = empty_f32_buf(&ctx, slots * n);
        let selected_out = empty_f32_buf(&ctx, slots * n);
        let expert_ids = shared_u32_buf(&ctx, &[expert as u32]);
        let n_buf = u32_buf(&ctx, n as u32);
        let k_buf = u32_buf(&ctx, k as u32);
        let per_expert_buf = u64_buf(&ctx, per_expert_bytes as u64);
        let shared_expert_id_buf = u32_buf(&ctx, u32::MAX);
        let sparse_off_buf = u64_buf(&ctx, sparse_page_off as u64);
        let shared_off_buf = u64_buf(&ctx, 0);
        let slots_buf = u32_buf(&ctx, slots as u32);
        let selected_weights = vec![selected_buf];
        let selected_offsets = vec![selected_page_off];

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_qwen_moe_decode_down_slots(
            &ctx,
            &enc,
            2,
            &sparse_buf,
            &sparse_buf,
            &input_buf,
            &full_out,
            &expert_ids,
            &n_buf,
            &k_buf,
            &per_expert_buf,
            &shared_expert_id_buf,
            &sparse_off_buf,
            &shared_off_buf,
            n,
            slots,
        );
        encode_qwen_moe_decode_q6k_selected_slots(
            &ctx,
            &enc,
            &selected_weights,
            &selected_offsets,
            &input_buf,
            &selected_out,
            &n_buf,
            &k_buf,
            &slots_buf,
            n,
            slots,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let full = readback(&full_out, slots * n);
        let selected = readback(&selected_out, slots * n);
        let _ = std::fs::remove_file(&path);

        let selected_norm: f32 = selected.iter().map(|v| v.abs()).sum();
        assert!(
            selected_norm > 1.0,
            "selected Q6_K output unexpectedly near zero"
        );
        for i in 0..slots * n {
            assert!(
                full[i].is_finite(),
                "full Q6_K arena output became non-finite at {i}: {} selected={}",
                full[i],
                selected[i]
            );
            let rel = (full[i] - selected[i]).abs() / selected[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} full={} selected={} rel={rel} sparse_page_off={} selected_page_off={} expert_start={}",
                full[i],
                selected[i],
                sparse_page_off,
                selected_page_off,
                expert_start
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_q4k_down_slots_use_slot_strided_activations_cpu_oracle() {
        use crate::tests_fixture;
        use rnb_cpu::quantize::quantize_q4_k_vec;

        let ctx = build_metal_context().expect("no metal device");
        let n = 7usize;
        let k = 512usize;
        let slots = 3usize;
        let per_expert_bytes = n * (k / 256) * 144;
        let weight: Vec<f32> = (0..n * k)
            .map(|i| {
                let row = i / k;
                let col = i % k;
                ((row * 29 + col * 17 + 11) % 101) as f32 * 0.00390625 - 0.1875
            })
            .collect();
        let weight_q4 = quantize_q4_k_vec(&weight);
        assert_eq!(weight_q4.len(), per_expert_bytes);
        let input: Vec<f32> = (0..slots)
            .flat_map(|slot| {
                (0..k).map(move |col| match slot {
                    0 => ((col * 13 + 3) % 47) as f32 * 0.015625 - 0.25,
                    1 => ((col * 19 + 7) % 53) as f32 * -0.01171875 + 0.375,
                    _ => ((col * 23 + 5) % 59) as f32 * 0.0078125 - 0.3125,
                })
            })
            .collect();
        let want = tests_fixture::q4k_gemm_reference(&weight_q4, n, k, &input, slots);
        for slot in 1..slots {
            let differs_from_slot0 = want[slot * n..(slot + 1) * n]
                .iter()
                .zip(&want[..n])
                .any(|(&value, &slot0)| (value - slot0).abs() > 1e-3);
            assert!(
                differs_from_slot0,
                "Q4_K down oracle must distinguish activation slot {slot} from slot 0"
            );
        }

        let weight_buf = shared_u8_buf(&ctx, &weight_q4);
        let input_buf = shared_f32_buf(&ctx, &input);
        let out_buf = empty_f32_buf(&ctx, slots * n);
        let expert_ids_buf = shared_u32_buf(&ctx, &vec![0u32; slots]);
        let n_buf = u32_buf(&ctx, n as u32);
        let k_buf = u32_buf(&ctx, k as u32);
        let per_expert_buf = u64_buf(&ctx, per_expert_bytes as u64);
        let shared_expert_id_buf = u32_buf(&ctx, u32::MAX);
        let sparse_off_buf = u64_buf(&ctx, 0);
        let shared_off_buf = u64_buf(&ctx, 0);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_qwen_moe_decode_down_slots(
            &ctx,
            &enc,
            0,
            &weight_buf,
            &weight_buf,
            &input_buf,
            &out_buf,
            &expert_ids_buf,
            &n_buf,
            &k_buf,
            &per_expert_buf,
            &shared_expert_id_buf,
            &sparse_off_buf,
            &shared_off_buf,
            n,
            slots,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let got = readback(&out_buf, slots * n);
        for slot in 0..slots {
            assert_global_rel_close(
                &format!("Q4_K direct down slot-strided activation slot {slot}"),
                &got[slot * n..(slot + 1) * n],
                &want[slot * n..(slot + 1) * n],
                1e-3,
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_q4k_pair_slots_matches_two_q4k_slots() {
        use rnb_cpu::quantize::quantize_q4_k_vec;

        let _env_lock = ENV_LOCK.lock().unwrap();
        let _pair_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_GATE_UP_PAIR", "1");
        let ctx = build_metal_context().expect("no metal device");
        let n_experts = 4usize;
        let shared_expert = n_experts as u32;
        let n = 64usize;
        let k = 512usize;
        let slots = 3usize;
        let per_expert_bytes = n * (k / 256) * 144;
        let expert_ids = [1u32, 3u32, shared_expert];

        let input: Vec<f32> = (0..k)
            .map(|i| ((i * 19 + 11) % 59) as f32 * 0.015625 - 0.35)
            .collect();
        let make_quant = |salt: usize| {
            let data: Vec<f32> = (0..n * k)
                .map(|i| {
                    let row = i / k;
                    let col = i % k;
                    ((row * 17 + col * 5 + salt) % 101) as f32 * 0.00390625 - 0.18
                })
                .collect();
            quantize_q4_k_vec(&data)
        };
        let mut gate_sparse = vec![0u8; n_experts * per_expert_bytes];
        let mut up_sparse = vec![0u8; n_experts * per_expert_bytes];
        for expert in [1usize, 3usize] {
            let start = expert * per_expert_bytes;
            gate_sparse[start..start + per_expert_bytes].copy_from_slice(&make_quant(7 + expert));
            up_sparse[start..start + per_expert_bytes].copy_from_slice(&make_quant(31 + expert));
        }
        let gate_shared = make_quant(71);
        let up_shared = make_quant(97);

        let gate_sparse_buf = shared_u8_buf(&ctx, &gate_sparse);
        let up_sparse_buf = shared_u8_buf(&ctx, &up_sparse);
        let gate_shared_buf = shared_u8_buf(&ctx, &gate_shared);
        let up_shared_buf = shared_u8_buf(&ctx, &up_shared);
        let input_buf = shared_f32_buf(&ctx, &input);
        let expert_ids_buf = shared_u32_buf(&ctx, &expert_ids);
        let n_buf = u32_buf(&ctx, n as u32);
        let k_buf = u32_buf(&ctx, k as u32);
        let per_buf = u64_buf(&ctx, per_expert_bytes as u64);
        let shared_id_buf = u32_buf(&ctx, shared_expert);
        let zero_buf = u32_buf(&ctx, 0);

        let gate_ref = empty_f32_buf(&ctx, slots * n);
        let up_ref = empty_f32_buf(&ctx, slots * n);
        let gate_pair = empty_f32_buf(&ctx, slots * n);
        let up_pair = empty_f32_buf(&ctx, slots * n);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_qwen_moe_decode_q4k_slots(
            &ctx,
            &enc,
            &gate_sparse_buf,
            &gate_shared_buf,
            &input_buf,
            &gate_ref,
            &expert_ids_buf,
            &n_buf,
            &k_buf,
            &per_buf,
            &shared_id_buf,
            &zero_buf,
            &zero_buf,
            n,
            slots,
        );
        encode_qwen_moe_decode_q4k_slots(
            &ctx,
            &enc,
            &up_sparse_buf,
            &up_shared_buf,
            &input_buf,
            &up_ref,
            &expert_ids_buf,
            &n_buf,
            &k_buf,
            &per_buf,
            &shared_id_buf,
            &zero_buf,
            &zero_buf,
            n,
            slots,
        );
        assert!(encode_qwen_moe_decode_q4k_pair_slots(
            &ctx,
            &enc,
            &gate_sparse_buf,
            &gate_shared_buf,
            &up_sparse_buf,
            &up_shared_buf,
            &input_buf,
            &gate_pair,
            &up_pair,
            &expert_ids_buf,
            &n_buf,
            &k_buf,
            &per_buf,
            &per_buf,
            &shared_id_buf,
            &zero_buf,
            &zero_buf,
            &zero_buf,
            &zero_buf,
            n,
            slots,
        ));
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let gate_ref = readback(&gate_ref, slots * n);
        let up_ref = readback(&up_ref, slots * n);
        let gate_pair = readback(&gate_pair, slots * n);
        let up_pair = readback(&up_pair, slots * n);

        for i in 0..slots * n {
            let gate_rel = (gate_pair[i] - gate_ref[i]).abs() / gate_ref[i].abs().max(1e-6);
            assert!(
                gate_rel < 1e-3,
                "gate mismatch i={i} pair={} ref={} rel={gate_rel}",
                gate_pair[i],
                gate_ref[i]
            );
            let up_rel = (up_pair[i] - up_ref[i]).abs() / up_ref[i].abs().max(1e-6);
            assert!(
                up_rel < 1e-3,
                "up mismatch i={i} pair={} ref={} rel={up_rel}",
                up_pair[i],
                up_ref[i]
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_q4k_selected_pair_slots_matches_two_selected_slots() {
        use rnb_cpu::quantize::quantize_q4_k_vec;

        let _env_lock = ENV_LOCK.lock().unwrap();
        let _pair_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_SELECTED_GATE_UP_PAIR", "1");
        let ctx = build_metal_context().expect("no metal device");
        let n = 64usize;
        let k = 512usize;
        let slots = 4usize;
        let prefix_len = 256usize;

        let input: Vec<f32> = (0..k)
            .map(|i| ((i * 23 + 17) % 67) as f32 * 0.0125 - 0.4)
            .collect();
        let make_quant = |salt: usize| {
            let data: Vec<f32> = (0..n * k)
                .map(|i| {
                    let row = i / k;
                    let col = i % k;
                    ((row * 11 + col * 13 + salt) % 103) as f32 * 0.003125 - 0.16
                })
                .collect();
            quantize_q4_k_vec(&data)
        };
        let make_prefixed = |quant: Vec<u8>, salt: u8| {
            let mut bytes = vec![salt; prefix_len];
            bytes.extend_from_slice(&quant);
            shared_u8_buf(&ctx, &bytes)
        };

        let gate_weights: Vec<_> = (0..slots)
            .map(|slot| make_prefixed(make_quant(5 + slot), 0xA0 + slot as u8))
            .collect();
        let up_weights: Vec<_> = (0..slots)
            .map(|slot| make_prefixed(make_quant(41 + slot), 0xC0 + slot as u8))
            .collect();
        let offsets = vec![prefix_len as u32; slots];

        let input_buf = shared_f32_buf(&ctx, &input);
        let n_buf = u32_buf(&ctx, n as u32);
        let k_buf = u32_buf(&ctx, k as u32);
        let slots_buf = u32_buf(&ctx, slots as u32);
        let gate_ref = empty_f32_buf(&ctx, slots * n);
        let up_ref = empty_f32_buf(&ctx, slots * n);
        let gate_pair = empty_f32_buf(&ctx, slots * n);
        let up_pair = empty_f32_buf(&ctx, slots * n);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_qwen_moe_decode_q4k_selected_slots(
            &ctx,
            &enc,
            &gate_weights,
            &offsets,
            &input_buf,
            &gate_ref,
            &n_buf,
            &k_buf,
            &slots_buf,
            n,
            slots,
        );
        encode_qwen_moe_decode_q4k_selected_slots(
            &ctx,
            &enc,
            &up_weights,
            &offsets,
            &input_buf,
            &up_ref,
            &n_buf,
            &k_buf,
            &slots_buf,
            n,
            slots,
        );
        assert!(encode_qwen_moe_decode_q4k_selected_pair_slots(
            &ctx,
            &enc,
            &gate_weights,
            &offsets,
            &up_weights,
            &offsets,
            &input_buf,
            &gate_pair,
            &up_pair,
            &n_buf,
            &k_buf,
            &slots_buf,
            n,
            slots,
        ));
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let gate_ref = readback(&gate_ref, slots * n);
        let up_ref = readback(&up_ref, slots * n);
        let gate_pair = readback(&gate_pair, slots * n);
        let up_pair = readback(&up_pair, slots * n);

        for i in 0..slots * n {
            let gate_rel = (gate_pair[i] - gate_ref[i]).abs() / gate_ref[i].abs().max(1e-6);
            assert!(
                gate_rel < 1e-3,
                "selected gate mismatch i={i} pair={} ref={} rel={gate_rel}",
                gate_pair[i],
                gate_ref[i]
            );
            let up_rel = (up_pair[i] - up_ref[i]).abs() / up_ref[i].abs().max(1e-6);
            assert!(
                up_rel < 1e-3,
                "selected up mismatch i={i} pair={} ref={} rel={up_rel}",
                up_pair[i],
                up_ref[i]
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_q4k_selected_slots_nsg2_matches_coalesced() {
        use rnb_cpu::quantize::quantize_q4_k_vec;

        let _env_lock = ENV_LOCK.lock().unwrap();
        let _nsg2_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_SELECTED_Q4K_NSG2", "1");
        let ctx = build_metal_context().expect("no metal device");
        let n = 66usize;
        let k = 512usize;
        let slots = 5usize;
        let prefix_len = 256usize;

        let input: Vec<f32> = (0..k)
            .map(|i| ((i * 7 + 29) % 71) as f32 * 0.01 - 0.32)
            .collect();
        let make_quant = |salt: usize| {
            let data: Vec<f32> = (0..n * k)
                .map(|i| {
                    let row = i / k;
                    let col = i % k;
                    ((row * 19 + col * 3 + salt) % 109) as f32 * 0.0025 - 0.14
                })
                .collect();
            quantize_q4_k_vec(&data)
        };
        let make_prefixed = |quant: Vec<u8>, salt: u8| {
            let mut bytes = vec![salt; prefix_len];
            bytes.extend_from_slice(&quant);
            shared_u8_buf(&ctx, &bytes)
        };
        let weights: Vec<_> = (0..slots)
            .map(|slot| make_prefixed(make_quant(13 + slot), 0x90 + slot as u8))
            .collect();
        let offsets = vec![prefix_len as u32; slots];

        let input_buf = shared_f32_buf(&ctx, &input);
        let n_buf = u32_buf(&ctx, n as u32);
        let k_buf = u32_buf(&ctx, k as u32);
        let slots_buf = u32_buf(&ctx, slots as u32);
        let ref_out = empty_f32_buf(&ctx, slots * n);
        let nsg2_out = empty_f32_buf(&ctx, slots * n);

        std::env::remove_var("RNB_METAL_QWEN35_MOE_SELECTED_Q4K_NSG2");
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_qwen_moe_decode_q4k_selected_slots(
            &ctx, &enc, &weights, &offsets, &input_buf, &ref_out, &n_buf, &k_buf, &slots_buf, n,
            slots,
        );
        std::env::set_var("RNB_METAL_QWEN35_MOE_SELECTED_Q4K_NSG2", "1");
        encode_qwen_moe_decode_q4k_selected_slots(
            &ctx, &enc, &weights, &offsets, &input_buf, &nsg2_out, &n_buf, &k_buf, &slots_buf, n,
            slots,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ref_out = readback(&ref_out, slots * n);
        let nsg2_out = readback(&nsg2_out, slots * n);
        for i in 0..slots * n {
            let rel = (nsg2_out[i] - ref_out[i]).abs() / ref_out[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "nsg2 mismatch i={i} nsg2={} ref={} rel={rel}",
                nsg2_out[i],
                ref_out[i]
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn residual_add_matches_cpu() {
        let ctx = build_metal_context().expect("no metal device");
        let dim = 3072usize;
        let hidden: Vec<f32> = (0..dim).map(|i| ((i % 19) as f32 - 9.0) * 0.3).collect();
        let down: Vec<f32> = (0..dim).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();

        let hidden_buf = shared_f32_buf(&ctx, &hidden);
        let down_buf = shared_f32_buf(&ctx, &down);
        let dim_buf = u32_buf(&ctx, dim as u32);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_residual_add(&ctx, &enc, &hidden_buf, &down_buf, &dim_buf, dim);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let gpu = readback(&hidden_buf, dim);
        for i in 0..dim {
            let expect = hidden[i] + down[i];
            let rel = (gpu[i] - expect).abs() / expect.abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} gpu={} expect={} rel={rel}",
                gpu[i],
                expect
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn hazard_tracking_serializes_norm_then_gemv() {
        use crate::compute::encode_gemv_q4k;
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");

        // K=256, N=4. norm 입력 dim=256(= GEMV K). norm weight=1.0 → normed=input/rms.
        let dim = 256usize;
        let n = 4usize;
        let input: Vec<f32> = (0..dim).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
        let norm_w = vec![1.0f32; dim];
        let eps = 1e-5f32;
        let single = tests_fixture::q4k_block_fixed();
        let weight: Vec<u8> = single.iter().cycle().take(n * 144).copied().collect();

        // reference: CPU rms_norm → host GEMV
        let cpu_normed = {
            let mean_sq: f32 = input.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let rms = (mean_sq + eps).sqrt();
            input.iter().map(|v| v / rms).collect::<Vec<f32>>()
        };
        let ref_out = crate::compute::gemv_q4k_with_ctx(&ctx, &weight, &cpu_normed, n, dim);

        // GPU: 단일 encoder 에 norm → gemv 연속 encode
        let in_buf = shared_f32_buf(&ctx, &input);
        let nw_buf = shared_f32_buf(&ctx, &norm_w);
        let normed_buf = empty_f32_buf(&ctx, dim);
        let dim_buf = u32_buf(&ctx, dim as u32);
        let eps_buf = f32_buf(&ctx, eps);
        let w_buf = shared_u8_buf(&ctx, &weight);
        let out_buf = empty_f32_buf(&ctx, n);
        let n_buf = u32_buf(&ctx, n as u32);
        let off_buf = u32_buf(&ctx, 0);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_rms_norm(
            &ctx,
            &enc,
            &in_buf,
            &nw_buf,
            &normed_buf,
            &dim_buf,
            &eps_buf,
        );
        encode_gemv_q4k(
            &ctx,
            &enc,
            &w_buf,
            &normed_buf,
            &out_buf,
            &n_buf,
            &dim_buf,
            &off_buf,
            n,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let gpu = readback(&out_buf, n);
        for i in 0..n {
            let rel = (gpu[i] - ref_out[i]).abs() / ref_out[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "hazard tracking 직렬화 실패? i={i} gpu={} ref={} rel={rel}",
                gpu[i],
                ref_out[i]
            );
        }
    }

    /// pm28: concurrent compute encoder 에서 독립 GEMV 그룹을 내부 barrier 없이 묶어도
    /// (그룹 앞뒤 `chain_barrier` 만) serial encoder 와 bit-identical. GDN carrier 의
    /// {qkv,gate,alpha,beta} 그룹 패턴(같은 normed 입력을 N GEMV 가 read, 각자 다른
    /// output 에 write — 독립) 검증. norm 완료 전 GEMV 가 stale read 하면(앞 barrier
    /// 누락) 틀린 값 → 20 회 반복으로 race 검출. serial 은 hazard tracking 자동 직렬화.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn concurrent_independent_gemv_group_matches_serial() {
        use crate::compute::{
            build_metal_context_with_opts, chain_barrier, chain_compute_encoder, encode_gemv_q4k,
            gemv_q4k_with_ctx,
        };
        use crate::tests_fixture;
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

        let dim = 256usize; // GEMV K = norm dim
        let n = 4usize; // GEMV N
        let n_groups = 4usize; // qkv/gate/alpha/beta 모사 (모두 normed 입력 read, 독립 output)
        let input: Vec<f32> = (0..dim).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
        let norm_w = vec![1.0f32; dim];
        let eps = 1e-5f32;
        let weight: Vec<u8> = tests_fixture::q4k_block_fixed()
            .iter()
            .cycle()
            .take(n * 144)
            .copied()
            .collect();

        // CPU reference: rms_norm → host GEMV.
        let cpu_normed = {
            let mean_sq: f32 = input.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let rms = (mean_sq + eps).sqrt();
            input.iter().map(|v| v / rms).collect::<Vec<f32>>()
        };

        // 같은 dispatch 로직을 ctx.chain_concurrent 에 따라 serial/concurrent encoder 로 실행.
        let run = |ctx: &MetalContext| -> Vec<Vec<f32>> {
            let in_buf = shared_f32_buf(ctx, &input);
            let nw_buf = shared_f32_buf(ctx, &norm_w);
            let normed_buf = empty_f32_buf(ctx, dim);
            let dim_buf = u32_buf(ctx, dim as u32);
            let eps_buf = f32_buf(ctx, eps);
            let n_buf = u32_buf(ctx, n as u32);
            let off_buf = u32_buf(ctx, 0);
            let w_buf = shared_u8_buf(ctx, &weight);
            let out_bufs: Vec<_> = (0..n_groups).map(|_| empty_f32_buf(ctx, n)).collect();

            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = chain_compute_encoder(ctx, &cmd);
            encode_rms_norm(ctx, &enc, &in_buf, &nw_buf, &normed_buf, &dim_buf, &eps_buf);
            chain_barrier(ctx, &enc); // norm 완료 보장 (concurrent 에서 필수)
            for out in &out_bufs {
                // 독립 그룹: 내부 barrier 없음 (서로 다른 output, normed read-only).
                encode_gemv_q4k(
                    ctx,
                    &enc,
                    &w_buf,
                    &normed_buf,
                    out,
                    &n_buf,
                    &dim_buf,
                    &off_buf,
                    n,
                );
            }
            chain_barrier(ctx, &enc); // 그룹 완료 보장
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            out_bufs.iter().map(|b| readback(b, n)).collect()
        };

        let ctx_serial = build_metal_context_with_opts(false, false).expect("no metal device");
        let ctx_conc = build_metal_context_with_opts(false, true).expect("no metal device");

        let cpu_ref = gemv_q4k_with_ctx(&ctx_serial, &weight, &cpu_normed, n, dim);

        for iter in 0..20 {
            let s = run(&ctx_serial);
            let c = run(&ctx_conc);
            for g in 0..n_groups {
                for i in 0..n {
                    assert_eq!(
                        s[g][i].to_bits(),
                        c[g][i].to_bits(),
                        "iter={iter} g={g} i={i} serial={} concurrent={} — concurrent race?",
                        s[g][i],
                        c[g][i]
                    );
                    let rel = (c[g][i] - cpu_ref[i]).abs() / cpu_ref[i].abs().max(1e-6);
                    assert!(
                        rel < 1e-3,
                        "iter={iter} g={g} i={i} concurrent={} cpu_ref={} rel={rel}",
                        c[g][i],
                        cpu_ref[i]
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn ffn_chain_matches_cpu_ffn() {
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");
        let hidden_dim = 256usize;
        let ffn_dim = 512usize;
        let eps = 1e-5f32;

        let hidden: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
            .collect();
        let norm_w: Vec<f32> = (0..hidden_dim)
            .map(|i| 1.0 + ((i % 5) as f32) * 0.01)
            .collect();

        // gate/up: rows=ffn_dim, K=hidden_dim → ffn_dim * (hidden_dim/256) blocks
        let single = tests_fixture::q4k_block_fixed();
        let gate_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(ffn_dim * (hidden_dim / 256) * 144)
            .copied()
            .collect();
        let up_w = gate_w.clone();
        // down: rows=hidden_dim, K=ffn_dim → hidden_dim * (ffn_dim/256) blocks
        let down_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(hidden_dim * (ffn_dim / 256) * 144)
            .copied()
            .collect();

        // ---- CPU reference FFN ----
        let cpu_out = {
            // rms_norm
            let mean_sq: f32 = hidden.iter().map(|v| v * v).sum::<f32>() / hidden_dim as f32;
            let rms = (mean_sq + eps).sqrt();
            let normed: Vec<f32> = (0..hidden_dim)
                .map(|i| (hidden[i] / rms) * norm_w[i])
                .collect();
            // gate/up GEMV (host 경로 재사용 — 동일 dequant 보장)
            let gate =
                crate::compute::gemv_q4k_with_ctx(&ctx, &gate_w, &normed, ffn_dim, hidden_dim);
            let up = crate::compute::gemv_q4k_with_ctx(&ctx, &up_w, &normed, ffn_dim, hidden_dim);
            // silu_mul
            let act: Vec<f32> = (0..ffn_dim)
                .map(|i| (gate[i] / (1.0 + (-gate[i]).exp())) * up[i])
                .collect();
            // down GEMV
            let down = crate::compute::gemv_q4k_with_ctx(&ctx, &down_w, &act, hidden_dim, ffn_dim);
            // residual
            (0..hidden_dim)
                .map(|i| hidden[i] + down[i])
                .collect::<Vec<f32>>()
        };

        // ---- GPU chain ----
        let carrier = FfnCarrier::new(&ctx, hidden_dim, ffn_dim, eps);
        let norm_w_buf = shared_f32_buf(&ctx, &norm_w);
        let gate_w_buf = shared_u8_buf(&ctx, &gate_w);
        let up_w_buf = shared_u8_buf(&ctx, &up_w);
        let down_w_buf = shared_u8_buf(&ctx, &down_w);
        let zero_off = u32_buf(&ctx, 0);

        let gpu_out = ffn_chain_dispatch(
            &ctx,
            &carrier,
            &hidden,
            &norm_w_buf,
            &gate_w_buf,
            &zero_off,
            &up_w_buf,
            &zero_off,
            &down_w_buf,
            &zero_off,
            false,
        );

        for i in 0..hidden_dim {
            let rel = (gpu_out[i] - cpu_out[i]).abs() / cpu_out[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} gpu={} cpu={} rel={rel}",
                gpu_out[i],
                cpu_out[i]
            );
        }
    }

    /// gate/up = Q4_K, down = Q6_K 인 chain 이 CPU reference FFN 과 일치하는지.
    /// 본선 dense 모델(qwen3.5-9B 등)의 실제 quant 조합(down 만 Q6_K).
    #[test]
    #[ignore = "requires a Metal device"]
    fn ffn_chain_q6k_down_matches_cpu_ffn() {
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");
        let hidden_dim = 256usize;
        let ffn_dim = 512usize;
        let eps = 1e-5f32;

        let hidden: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
            .collect();
        // norm_w 를 작게 — gate/up 이 silu 포화 영역(act≈0) 으로 가지 않도록.
        // 작은 norm_w → 작은 gate → silu 선형 영역 → act/down 기여가 의미있게 남음.
        let norm_w: Vec<f32> = (0..hidden_dim)
            .map(|i| 0.01 + ((i % 5) as f32) * 0.001)
            .collect();

        // gate/up: Q4_K, rows=ffn_dim, K=hidden_dim
        let q4k = tests_fixture::q4k_block_fixed();
        let gate_w: Vec<u8> = q4k
            .iter()
            .cycle()
            .take(ffn_dim * (hidden_dim / 256) * 144)
            .copied()
            .collect();
        let up_w = gate_w.clone();
        // down: Q6_K, rows=hidden_dim, K=ffn_dim → hidden_dim * (ffn_dim/256) blocks × 210B
        let q6k = tests_fixture::q6k_block_fixed();
        let down_w: Vec<u8> = q6k
            .iter()
            .cycle()
            .take(hidden_dim * (ffn_dim / 256) * 210)
            .copied()
            .collect();

        // ---- CPU reference FFN ----
        let cpu_out = {
            let mean_sq: f32 = hidden.iter().map(|v| v * v).sum::<f32>() / hidden_dim as f32;
            let rms = (mean_sq + eps).sqrt();
            let normed: Vec<f32> = (0..hidden_dim)
                .map(|i| (hidden[i] / rms) * norm_w[i])
                .collect();
            // gate/up: host Q4_K 경로(동일 dequant 보장)
            let gate =
                crate::compute::gemv_q4k_with_ctx(&ctx, &gate_w, &normed, ffn_dim, hidden_dim);
            let up = crate::compute::gemv_q4k_with_ctx(&ctx, &up_w, &normed, ffn_dim, hidden_dim);
            let act: Vec<f32> = (0..ffn_dim)
                .map(|i| (gate[i] / (1.0 + (-gate[i]).exp())) * up[i])
                .collect();
            // down: Q6_K — 모든 block 이 동일 fixture → row weight = dequant 반복.
            let down_blk_w = tests_fixture::q6k_dequant(&q6k);
            let down: Vec<f32> = (0..hidden_dim)
                .map(|_| {
                    (0..ffn_dim)
                        .map(|c| down_blk_w[c % 256] * act[c])
                        .sum::<f32>()
                })
                .collect();
            (0..hidden_dim)
                .map(|i| hidden[i] + down[i])
                .collect::<Vec<f32>>()
        };

        // ---- GPU chain (down_is_q6k = true) ----
        let carrier = FfnCarrier::new(&ctx, hidden_dim, ffn_dim, eps);
        let norm_w_buf = shared_f32_buf(&ctx, &norm_w);
        let gate_w_buf = shared_u8_buf(&ctx, &gate_w);
        let up_w_buf = shared_u8_buf(&ctx, &up_w);
        let down_w_buf = shared_u8_buf(&ctx, &down_w);
        let zero_off = u32_buf(&ctx, 0);

        let gpu_out = ffn_chain_dispatch(
            &ctx,
            &carrier,
            &hidden,
            &norm_w_buf,
            &gate_w_buf,
            &zero_off,
            &up_w_buf,
            &zero_off,
            &down_w_buf,
            &zero_off,
            true,
        );

        for i in 0..hidden_dim {
            let rel = (gpu_out[i] - cpu_out[i]).abs() / cpu_out[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} gpu={} cpu={} rel={rel}",
                gpu_out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn gemv_q6k_device_inout_matches_cpu_dequant() {
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");
        // N=4 rows, K=512 (2 blocks per row) — multi-block offset 도 검증.
        let n = 4usize;
        let k = 512usize;
        let num_blocks = k / 256;
        let single = tests_fixture::q6k_block_fixed();
        let weight: Vec<u8> = single
            .iter()
            .cycle()
            .take(n * num_blocks * 210)
            .copied()
            .collect();
        let input: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();

        // CPU ground truth: 모든 block 이 동일 fixture → 각 row 의 weight 는
        // 2 블록(각 256) dequant 를 이어붙인 K=512 벡터. row 무관(동일 weight) 이라
        // 모든 row 동일 결과.
        let blk_w = tests_fixture::q6k_dequant(&single);
        let host: Vec<f32> = (0..n)
            .map(|_| (0..k).map(|j| blk_w[j % 256] * input[j]).sum::<f32>())
            .collect();

        // device-in/out 경로: 모든 buffer 를 미리 만들고 encode 만.
        let w_buf = shared_u8_buf(&ctx, &weight);
        let in_buf = shared_f32_buf(&ctx, &input);
        let out_buf = empty_f32_buf(&ctx, n);
        let n_buf = u32_buf(&ctx, n as u32);
        let k_buf = u32_buf(&ctx, k as u32);
        let off_buf = u32_buf(&ctx, 0);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        crate::compute::encode_gemv_q6k(
            &ctx, &enc, &w_buf, &in_buf, &out_buf, &n_buf, &k_buf, &off_buf, n,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let dev = readback(&out_buf, n);
        for i in 0..n {
            let rel = (dev[i] - host[i]).abs() / host[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} dev={} host={} rel={rel}",
                dev[i],
                host[i]
            );
        }
    }

    /// QKV chain(단일 command buffer 3 GEMV)이 개별 GEMV 와 동일 결과.
    #[test]
    #[ignore = "requires a Metal device"]
    fn qkv_chain_matches_individual() {
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");
        let hidden_dim = 256usize;
        let q_out_dim = 512usize; // q (heads*head_dim)
        let kv_dim = 256usize; // k/v (GQA)

        let norm: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
            .collect();
        let single = tests_fixture::q4k_block_fixed();
        let nb = hidden_dim / 256;
        let q_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(q_out_dim * nb * 144)
            .copied()
            .collect();
        let k_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(kv_dim * nb * 144)
            .copied()
            .collect();
        // v 는 다른 패턴이 아니어도 같은 weight 면 k 와 동일 결과 — 독립성만 검증되면 됨.
        let v_w = k_w.clone();

        // 개별 GEMV reference (host 경로 — 동일 dequant)
        let q_ref = crate::compute::gemv_q4k_with_ctx(&ctx, &q_w, &norm, q_out_dim, hidden_dim);
        let k_ref = crate::compute::gemv_q4k_with_ctx(&ctx, &k_w, &norm, kv_dim, hidden_dim);
        let v_ref = crate::compute::gemv_q4k_with_ctx(&ctx, &v_w, &norm, kv_dim, hidden_dim);

        // QKV chain
        let carrier = QkvCarrier::new(&ctx, hidden_dim, q_out_dim, kv_dim);
        let q_w_buf = shared_u8_buf(&ctx, &q_w);
        let k_w_buf = shared_u8_buf(&ctx, &k_w);
        let v_w_buf = shared_u8_buf(&ctx, &v_w);
        let zero_off = u32_buf(&ctx, 0);
        let (q, k, v) = qkv_chain_dispatch(
            &ctx, &carrier, &norm, &q_w_buf, &zero_off, &k_w_buf, &zero_off, &v_w_buf, &zero_off,
        );

        let check = |got: &[f32], want: &[f32], name: &str| {
            for i in 0..want.len() {
                let rel = (got[i] - want[i]).abs() / want[i].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "{name} i={i} got={} want={} rel={rel}",
                    got[i],
                    want[i]
                );
            }
        };
        check(&q, &q_ref, "q");
        check(&k, &k_ref, "k");
        check(&v, &v_ref, "v");
    }

    /// GDN qkv+gate chain(단일 command buffer 2 GEMV)이 개별 GEMV 와 일치.
    #[test]
    #[ignore = "requires a Metal device"]
    fn gdn_inproj_chain_matches_individual() {
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");
        let hidden_dim = 256usize;
        let qkv_dim = 768usize; // conv_channels (d_inner + 2*n_group*d_state)
        let gate_dim = 512usize; // d_inner

        let norm: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
            .collect();
        let single = tests_fixture::q4k_block_fixed();
        let nb = hidden_dim / 256;
        let qkv_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(qkv_dim * nb * 144)
            .copied()
            .collect();
        let gate_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(gate_dim * nb * 144)
            .copied()
            .collect();

        let qkv_ref = crate::compute::gemv_q4k_with_ctx(&ctx, &qkv_w, &norm, qkv_dim, hidden_dim);
        let gate_ref =
            crate::compute::gemv_q4k_with_ctx(&ctx, &gate_w, &norm, gate_dim, hidden_dim);

        let carrier = GdnInprojCarrier::new(&ctx, hidden_dim, qkv_dim, gate_dim);
        let qkv_w_buf = shared_u8_buf(&ctx, &qkv_w);
        let gate_w_buf = shared_u8_buf(&ctx, &gate_w);
        let zero_off = u32_buf(&ctx, 0);
        let (qkv, gate) = gdn_inproj_chain_dispatch(
            &ctx,
            &carrier,
            &norm,
            &qkv_w_buf,
            &zero_off,
            &gate_w_buf,
            &zero_off,
        );

        let check = |got: &[f32], want: &[f32], name: &str| {
            for i in 0..want.len() {
                let rel = (got[i] - want[i]).abs() / want[i].abs().max(1e-6);
                assert!(
                    rel < 1e-3,
                    "{name} i={i} got={} want={} rel={rel}",
                    got[i],
                    want[i]
                );
            }
        };
        check(&qkv, &qkv_ref, "qkv");
        check(&gate, &gate_ref, "gate");
    }

    /// attention O chain(o_proj + residual 단일 command buffer)이 개별 연산과 일치.
    #[test]
    #[ignore = "requires a Metal device"]
    fn o_chain_matches_individual() {
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");
        let hidden_dim = 256usize;
        let q_dim = 512usize; // attention output dim (heads*head_dim)

        let attn_out: Vec<f32> = (0..q_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.04).collect();
        let hidden: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.06)
            .collect();
        let single = tests_fixture::q4k_block_fixed();
        // o_weight: rows=hidden_dim, K=q_dim → hidden_dim * (q_dim/256) blocks
        let o_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(hidden_dim * (q_dim / 256) * 144)
            .copied()
            .collect();

        // 개별 reference: o_proj GEMV(attn → proj) + residual(hidden + proj)
        let proj = crate::compute::gemv_q4k_with_ctx(&ctx, &o_w, &attn_out, hidden_dim, q_dim);
        let want: Vec<f32> = (0..hidden_dim).map(|i| hidden[i] + proj[i]).collect();

        // O chain
        let carrier = OChainCarrier::new(&ctx, hidden_dim, q_dim);
        let o_w_buf = shared_u8_buf(&ctx, &o_w);
        let zero_off = u32_buf(&ctx, 0);
        let got = o_chain_dispatch(&ctx, &carrier, &attn_out, &hidden, &o_w_buf, &zero_off);

        for i in 0..hidden_dim {
            let rel = (got[i] - want[i]).abs() / want[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "i={i} got={} want={} rel={rel}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn ffn_chain_carrier_reuse_deterministic() {
        use crate::tests_fixture;
        let ctx = build_metal_context().expect("no metal device");
        let (hidden_dim, ffn_dim, eps) = (256usize, 512usize, 1e-5f32);
        let hidden: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
            .collect();
        let norm_w = vec![1.0f32; hidden_dim];
        let single = tests_fixture::q4k_block_fixed();
        let gate_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(ffn_dim * (hidden_dim / 256) * 144)
            .copied()
            .collect();
        let down_w: Vec<u8> = single
            .iter()
            .cycle()
            .take(hidden_dim * (ffn_dim / 256) * 144)
            .copied()
            .collect();

        let carrier = FfnCarrier::new(&ctx, hidden_dim, ffn_dim, eps);
        let norm_w_buf = shared_f32_buf(&ctx, &norm_w);
        let gate_w_buf = shared_u8_buf(&ctx, &gate_w);
        let down_w_buf = shared_u8_buf(&ctx, &down_w);
        let zero_off = u32_buf(&ctx, 0);

        let r1 = ffn_chain_dispatch(
            &ctx,
            &carrier,
            &hidden,
            &norm_w_buf,
            &gate_w_buf,
            &zero_off,
            &gate_w_buf,
            &zero_off,
            &down_w_buf,
            &zero_off,
            false,
        );
        let r2 = ffn_chain_dispatch(
            &ctx,
            &carrier,
            &hidden,
            &norm_w_buf,
            &gate_w_buf,
            &zero_off,
            &gate_w_buf,
            &zero_off,
            &down_w_buf,
            &zero_off,
            false,
        );
        assert_eq!(r1, r2, "carrier 재사용 시 동일 입력 → 동일 출력이어야 함");
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_id_partial_reduce_scatter_is_deterministic() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _primitive_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", "1");
        let ctx = build_metal_context().expect("no metal device");
        let ffn_tiles = 3usize;
        let tile_slots = 2usize;
        let hidden_tile = 4usize;
        let hidden_dim = 8usize;
        let hidden0 = 2usize;
        let group_start = 0u32;
        let token_ids = [1u32, 0u32];
        let route_weights = [0.25f32, 0.5f32];
        let partial = [
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0,
            2.0, 0.0, -1.0, 1.0, 1.0, -2.0, 3.0, -4.0,
        ];

        let partial_buf = shared_f32_buf(&ctx, &partial);
        let out_init = vec![0.0f32; 2 * hidden_dim];
        let out_buf = shared_f32_buf(&ctx, &out_init);
        let token_ids_buf = shared_u32_buf(&ctx, &token_ids);
        let route_weights_buf = shared_f32_buf(&ctx, &route_weights);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        crate::compute::encode_qwen_moe_id_partial_reduce_scatter(
            &ctx,
            &enc,
            &partial_buf,
            &out_buf,
            &token_ids_buf,
            &route_weights_buf,
            ffn_tiles,
            tile_slots,
            hidden_tile,
            hidden_dim,
            hidden0,
            group_start,
        )
        .expect("encode reduce-scatter");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let got = readback(&out_buf, 2 * hidden_dim);
        let want = vec![
            0.0, 0.0, 8.0, 14.0, 24.0, 28.0, 0.0, 0.0, 0.0, 0.0, 0.875, 0.75, 0.875, 1.75, 0.0, 0.0,
        ];
        for i in 0..want.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-6,
                "i={i} got={} want={}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_id_reduce_scatter_preserves_serial_expert_order() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _primitive_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", "1");
        let ctx = build_metal_context().expect("no metal device");
        let token_ids = [0u32, 0, 0];
        let route_weights = [1.0f32, 1.0, 1.0];
        let partials = [
            shared_f32_buf(&ctx, &[1.0e20f32]),
            shared_f32_buf(&ctx, &[-1.0e20f32]),
            shared_f32_buf(&ctx, &[3.0f32]),
        ];
        let out_buf = shared_f32_buf(&ctx, &[0.0f32]);
        let token_ids_buf = shared_u32_buf(&ctx, &token_ids);
        let route_weights_buf = shared_f32_buf(&ctx, &route_weights);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        for (group_start, partial_buf) in partials.iter().enumerate() {
            crate::compute::encode_qwen_moe_id_partial_reduce_scatter(
                &ctx,
                &enc,
                partial_buf,
                &out_buf,
                &token_ids_buf,
                &route_weights_buf,
                1,
                1,
                1,
                1,
                0,
                group_start as u32,
            )
            .expect("encode serial reduce-scatter");
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let got = readback(&out_buf, 1);
        assert_eq!(got[0], 3.0);
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_id_q4_gate_up_tile_matches_existing_id_gate_up_slice() {
        use rnb_cpu::quantize::quantize_q4_k_vec;

        let _env_lock = ENV_LOCK.lock().unwrap();
        let _primitive_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", "1");
        let _id_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_ID_MATMUL", "1");
        let ctx = build_metal_context().expect("no metal device");
        let n = 128usize;
        let k = 512usize;
        let slots = 3usize;
        let prefix_len = 256usize;
        let ffn_tile0 = 64usize;
        let ffn_tile = 64usize;
        let input: Vec<f32> = (0..slots * k)
            .map(|i| ((i * 7 + 19) % 97) as f32 * 0.00625 - 0.31)
            .collect();
        let make_weights = |salt: usize| -> Vec<f32> {
            (0..n * k)
                .map(|i| {
                    let row = i / k;
                    let col = i % k;
                    ((row * 11 + col * 13 + salt) % 101) as f32 * 0.003125 - 0.15
                })
                .collect()
        };
        let make_raw = |weights: Vec<f32>, fill: u8| {
            let mut raw = vec![fill; prefix_len];
            raw.extend_from_slice(&quantize_q4_k_vec(&weights));
            raw
        };
        let gate_raw = make_raw(make_weights(5), 0xA5);
        let up_raw = make_raw(make_weights(41), 0xC3);

        let input_buf = shared_f32_buf(&ctx, &input);
        let gate_raw_buf = shared_u8_buf(&ctx, &gate_raw);
        let up_raw_buf = shared_u8_buf(&ctx, &up_raw);
        let token_ids = shared_u32_buf(&ctx, &[0, 1, 2]);
        let expert_offsets = shared_u32_buf(&ctx, &[0]);
        let expert_counts = shared_u32_buf(&ctx, &[slots as u32]);
        let block_experts = shared_u32_buf(&ctx, &[0]);
        let block_local0 = shared_u32_buf(&ctx, &[0]);
        let gate_full = empty_f32_buf(&ctx, slots * n);
        let up_full = empty_f32_buf(&ctx, slots * n);
        let gate_tile = empty_f32_buf(&ctx, slots * ffn_tile);
        let up_tile = empty_f32_buf(&ctx, slots * ffn_tile);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        crate::compute::encode_gemm_q4k_tensorops_id(
            &ctx,
            &enc,
            &gate_raw_buf,
            prefix_len as u32,
            &input_buf,
            &gate_full,
            &token_ids,
            &expert_offsets,
            &expert_counts,
            &block_experts,
            &block_local0,
            n,
            k,
            1,
            gate_raw.len() - prefix_len,
        );
        crate::compute::encode_gemm_q4k_tensorops_id(
            &ctx,
            &enc,
            &up_raw_buf,
            prefix_len as u32,
            &input_buf,
            &up_full,
            &token_ids,
            &expert_offsets,
            &expert_counts,
            &block_experts,
            &block_local0,
            n,
            k,
            1,
            up_raw.len() - prefix_len,
        );
        crate::compute::encode_qwen_moe_id_q4_gate_up_tile(
            &ctx,
            &enc,
            &gate_raw_buf,
            prefix_len as u32,
            &up_raw_buf,
            prefix_len as u32,
            &input_buf,
            &gate_tile,
            &up_tile,
            &token_ids,
            &expert_offsets,
            &expert_counts,
            &block_experts,
            &block_local0,
            n,
            k,
            1,
            gate_raw.len() - prefix_len,
            up_raw.len() - prefix_len,
            ffn_tile0,
            ffn_tile,
        )
        .expect("encode q4 gate/up tile");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let gate_full = readback(&gate_full, slots * n);
        let up_full = readback(&up_full, slots * n);
        let gate_tile = readback(&gate_tile, slots * ffn_tile);
        let up_tile = readback(&up_tile, slots * ffn_tile);
        for slot in 0..slots {
            for col in 0..ffn_tile {
                let gate_got = gate_tile[slot * ffn_tile + col];
                let gate_want = gate_full[slot * n + ffn_tile0 + col];
                let gate_rel = (gate_got - gate_want).abs() / gate_want.abs().max(1e-6);
                assert!(
                    gate_rel < 1e-3,
                    "gate slot={slot} col={col} got={gate_got} want={gate_want} rel={gate_rel}"
                );
                let up_got = up_tile[slot * ffn_tile + col];
                let up_want = up_full[slot * n + ffn_tile0 + col];
                let up_rel = (up_got - up_want).abs() / up_want.abs().max(1e-6);
                assert!(
                    up_rel < 1e-3,
                    "up slot={slot} col={col} got={up_got} want={up_want} rel={up_rel}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_id_q6_down_partial_matches_existing_id_down_slice() {
        use crate::tests_fixture;

        let _env_lock = ENV_LOCK.lock().unwrap();
        let _primitive_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_ID_PRIMITIVE", "1");
        let ctx = build_metal_context().expect("no metal device");
        let hidden_dim = 96usize;
        let hidden0 = 32usize;
        let hidden_tile = 64usize;
        let ffn_dim = 512usize;
        let ffn_tile0 = 256usize;
        let ffn_tile = 256usize;
        let ffn_tile_idx = 1usize;
        let slots = 2usize;
        let prefix_len = 256usize;
        let act: Vec<f32> = (0..slots * ffn_tile)
            .map(|i| ((i * 13 + 3) % 89) as f32 * 0.004 - 0.18)
            .collect();
        let mut full_act = vec![0.0f32; slots * ffn_dim];
        for slot in 0..slots {
            let dst = slot * ffn_dim + ffn_tile0;
            let src = slot * ffn_tile;
            full_act[dst..dst + ffn_tile].copy_from_slice(&act[src..src + ffn_tile]);
        }
        let weights = tests_fixture::q6k_rows_pattern(hidden_dim, ffn_dim);
        let cpu =
            tests_fixture::q6k_gemm_reference(&weights, hidden_dim, ffn_dim, &full_act, slots);
        let mut raw = vec![0x5Au8; prefix_len];
        raw.extend_from_slice(&weights);

        let act_buf = shared_f32_buf(&ctx, &act);
        let raw_buf = shared_u8_buf(&ctx, &raw);
        let partial_len = (ffn_tile_idx + 1) * slots * hidden_tile;
        let partial = empty_f32_buf(&ctx, partial_len);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        crate::compute::encode_qwen_moe_id_q6_down_partial_tile(
            &ctx,
            &enc,
            &raw_buf,
            prefix_len as u32,
            &act_buf,
            &partial,
            hidden_dim,
            hidden0,
            hidden_tile,
            ffn_dim,
            ffn_tile0,
            ffn_tile,
            slots,
            ffn_tile_idx,
        )
        .expect("encode q6 down partial tile");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let got = readback(&partial, partial_len);
        for slot in 0..slots {
            for row in 0..hidden_tile {
                let want = cpu[slot * hidden_dim + hidden0 + row];
                let got = got[(ffn_tile_idx * slots + slot) * hidden_tile + row];
                let rel = (got - want).abs() / want.abs().max(1e-5);
                assert!(
                    rel < 2e-2,
                    "slot={slot} row={row} got={got} want={want} rel={rel}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v3_token_rank_pipelines_compile() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _v3_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V3", "1");
        let ctx = build_metal_context().expect("no metal device");
        assert!(ctx.gemm_q4k_tensorops_id_v2_f16_pipeline.is_some());
        assert!(ctx.qwen_moe_v3_q6_down_pipeline.is_some());
        assert!(ctx.qwen_moe_v3_q4_down_pipeline.is_some());
        assert!(ctx.qwen_moe_v3_token_rank_combine_pipeline.is_some());
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v4_rank_down_scatter_pipelines_compile() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _v4_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4", "1");
        let ctx = build_metal_context().expect("no metal device");
        assert!(ctx.gemm_q4k_tensorops_id_v2_f16_pipeline.is_some());
        assert!(ctx.qwen_moe_v4_q6_down_scatter_pipeline.is_some());
        assert!(ctx.qwen_moe_v4_q4_down_scatter_pipeline.is_some());
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v4_supported_requires_gate_up_silu_and_down_scatter() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _v4_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4", "1");
        let ctx = build_metal_context().expect("no metal device");
        assert!(qwen_moe_prefill_mulmmid_v4_supported(&ctx, false));
        assert!(qwen_moe_prefill_mulmmid_v4_supported(&ctx, true));
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v4_down_scatter_wrappers_encode_q4_and_q6() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _v4_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4", "1");
        let ctx = build_metal_context().expect("no metal device");
        let n = 64usize;
        let k = 256usize;
        let q4_stride = n * (k / 256) * 144;
        let q6_stride = n * (k / 256) * 210;
        let q4_raw = shared_u8_buf(&ctx, &vec![0u8; q4_stride]);
        let q6_raw = shared_u8_buf(&ctx, &vec![0u8; q6_stride]);
        let act = empty_f16_buf(&ctx, k);
        let out = empty_f32_buf(&ctx, n);
        let token_ids = shared_u32_buf(&ctx, &[0]);
        let routes = shared_f32_buf(&ctx, &[1.0]);
        let offsets = shared_u32_buf(&ctx, &[0]);
        let counts = shared_u32_buf(&ctx, &[1]);
        let block_experts = shared_u32_buf(&ctx, &[0]);
        let block_local0 = shared_u32_buf(&ctx, &[0]);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        crate::compute::encode_qwen_moe_v4_q4_down_scatter(
            &ctx,
            &enc,
            &q4_raw,
            0,
            &act,
            &out,
            &token_ids,
            &routes,
            &offsets,
            &counts,
            &block_experts,
            &block_local0,
            0,
            1,
            0,
            1,
            n,
            k,
            q4_stride,
        )
        .expect("encode q4 v4 down scatter");
        crate::compute::encode_qwen_moe_v4_q6_down_scatter(
            &ctx,
            &enc,
            &q6_raw,
            0,
            &act,
            &out,
            &token_ids,
            &routes,
            &offsets,
            &counts,
            &block_experts,
            &block_local0,
            0,
            1,
            0,
            1,
            n,
            k,
            q6_stride,
        )
        .expect("encode q6 v4 down scatter");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v4_q4_down_scatter_matches_cpu_rank_oracle() {
        use crate::tests_fixture;

        let _env_lock = ENV_LOCK.lock().unwrap();
        let _v4_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4", "1");
        let ctx = build_metal_context().expect("no metal device");
        let n = 64usize;
        let k = 256usize;
        let seq_len = 2usize;
        let n_expert = 2usize;
        let n_expert_used = 2usize;
        let expert_ids = [1, 0, 0, 1];
        let token_ids = [0, 0, 1, 1];
        let route_weights = [0.25, 0.5, 0.75, 1.25];
        let plan = build_qwen_moe_mul_mm_id_v4_plan_token_major(
            &expert_ids,
            &token_ids,
            &route_weights,
            n_expert,
            seq_len,
            n_expert_used,
            64,
        )
        .expect("v4 plan");
        let weight = q4_rows_pattern(n_expert * n, k);
        let expert_stride = n * (k / 256) * 144;
        let act_values = (0..plan.token_ids_sorted.len() * k)
            .map(|i| ((i * 7 + 3) % 97) as f32 * 0.0035 - 0.16)
            .collect::<Vec<_>>();
        let (act_bytes, act_cpu) = half_bytes_and_cpu(&act_values);
        let want = qwen_moe_v4_rank_scatter_reference(
            &weight,
            n,
            k,
            seq_len,
            n_expert,
            n_expert_used,
            144,
            &plan,
            &act_cpu,
            tests_fixture::q4k_dequant,
        );

        let weight_buf = shared_u8_buf(&ctx, &weight);
        let act_buf = shared_u8_buf(&ctx, &act_bytes);
        let out_buf = shared_f32_buf(&ctx, &vec![0.0f32; seq_len * n]);
        let token_ids_buf = shared_u32_buf(&ctx, &plan.token_ids_sorted);
        let routes_buf = shared_f32_buf(&ctx, &plan.route_weights_sorted);
        let expert_rank_offsets = shared_u32_buf(&ctx, &plan.expert_rank_offsets);
        let expert_rank_counts = shared_u32_buf(&ctx, &plan.expert_rank_counts);
        let rank_block_experts = shared_u32_buf(&ctx, &plan.rank_block_experts);
        let rank_block_local0 = shared_u32_buf(&ctx, &plan.rank_block_local0);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        for rank in 0..n_expert_used {
            crate::compute::encode_qwen_moe_v4_q4_down_scatter(
                &ctx,
                &enc,
                &weight_buf,
                0,
                &act_buf,
                &out_buf,
                &token_ids_buf,
                &routes_buf,
                &expert_rank_offsets,
                &expert_rank_counts,
                &rank_block_experts,
                &rank_block_local0,
                plan.rank_block_offsets[rank] as usize,
                plan.rank_block_counts[rank] as usize,
                rank,
                n_expert_used,
                n,
                k,
                expert_stride,
            )
            .expect("encode v4 q4 down scatter");
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let got = readback(&out_buf, seq_len * n);
        assert_global_rel_close("v4 q4 down scatter", &got, &want, 2e-3);
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v4_q6_down_scatter_matches_cpu_rank_oracle() {
        use crate::tests_fixture;

        let _env_lock = ENV_LOCK.lock().unwrap();
        let _v4_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4", "1");
        let ctx = build_metal_context().expect("no metal device");
        let n = 64usize;
        let k = 256usize;
        let seq_len = 2usize;
        let n_expert = 2usize;
        let n_expert_used = 2usize;
        let expert_ids = [1, 0, 0, 1];
        let token_ids = [0, 0, 1, 1];
        let route_weights = [0.25, 0.5, 0.75, 1.25];
        let plan = build_qwen_moe_mul_mm_id_v4_plan_token_major(
            &expert_ids,
            &token_ids,
            &route_weights,
            n_expert,
            seq_len,
            n_expert_used,
            64,
        )
        .expect("v4 plan");
        let weight = tests_fixture::q6k_rows_pattern(n_expert * n, k);
        let expert_stride = n * (k / 256) * 210;
        let act_values = (0..plan.token_ids_sorted.len() * k)
            .map(|i| ((i * 5 + 11) % 89) as f32 * 0.003 - 0.13)
            .collect::<Vec<_>>();
        let (act_bytes, act_cpu) = half_bytes_and_cpu(&act_values);
        let want = qwen_moe_v4_rank_scatter_reference(
            &weight,
            n,
            k,
            seq_len,
            n_expert,
            n_expert_used,
            210,
            &plan,
            &act_cpu,
            tests_fixture::q6k_dequant,
        );

        let weight_buf = shared_u8_buf(&ctx, &weight);
        let act_buf = shared_u8_buf(&ctx, &act_bytes);
        let out_buf = shared_f32_buf(&ctx, &vec![0.0f32; seq_len * n]);
        let token_ids_buf = shared_u32_buf(&ctx, &plan.token_ids_sorted);
        let routes_buf = shared_f32_buf(&ctx, &plan.route_weights_sorted);
        let expert_rank_offsets = shared_u32_buf(&ctx, &plan.expert_rank_offsets);
        let expert_rank_counts = shared_u32_buf(&ctx, &plan.expert_rank_counts);
        let rank_block_experts = shared_u32_buf(&ctx, &plan.rank_block_experts);
        let rank_block_local0 = shared_u32_buf(&ctx, &plan.rank_block_local0);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        for rank in 0..n_expert_used {
            crate::compute::encode_qwen_moe_v4_q6_down_scatter(
                &ctx,
                &enc,
                &weight_buf,
                0,
                &act_buf,
                &out_buf,
                &token_ids_buf,
                &routes_buf,
                &expert_rank_offsets,
                &expert_rank_counts,
                &rank_block_experts,
                &rank_block_local0,
                plan.rank_block_offsets[rank] as usize,
                plan.rank_block_counts[rank] as usize,
                rank,
                n_expert_used,
                n,
                k,
                expert_stride,
            )
            .expect("encode v4 q6 down scatter");
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let got = readback(&out_buf, seq_len * n);
        assert_global_rel_close("v4 q6 down scatter", &got, &want, 2e-3);
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v4_chain_encoder_runs_gate_up_silu_and_rank_down_scatter() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _v4_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4", "1");
        let ctx = build_metal_context().expect("no metal device");
        let hidden_dim = 256usize;
        let ffn_dim = 256usize;
        let seq_len = 1usize;
        let n_expert_used = 1usize;
        let n_expert = 1usize;
        let plan = build_qwen_moe_mul_mm_id_v4_plan_token_major(
            &[0],
            &[0],
            &[1.0],
            n_expert,
            seq_len,
            n_expert_used,
            64,
        )
        .expect("v4 plan");
        let id_block_count = plan.id_block_experts.len();
        let q4_stride = ffn_dim * (hidden_dim / 256) * 144;
        let down_q4_stride = hidden_dim * (ffn_dim / 256) * 144;
        let down_q6_stride = hidden_dim * (ffn_dim / 256) * 210;
        let gate_raw = shared_u8_buf(&ctx, &vec![0u8; q4_stride * n_expert]);
        let up_raw = shared_u8_buf(&ctx, &vec![0u8; q4_stride * n_expert]);
        let down_q4_raw = shared_u8_buf(&ctx, &vec![0u8; down_q4_stride * n_expert]);
        let down_q6_raw = shared_u8_buf(&ctx, &vec![0u8; down_q6_stride * n_expert]);
        let accum = QwenMoePrefillAccumCarrier::new(&ctx, hidden_dim, seq_len);
        accum.upload_norm_all(&vec![0.0f32; hidden_dim * seq_len]);
        let carrier = QwenMoeMulMmIdV4Carrier::new(
            &ctx,
            seq_len,
            n_expert_used,
            hidden_dim,
            ffn_dim,
            n_expert,
            id_block_count,
            64 * 1024 * 1024,
        )
        .expect("v4 carrier");
        let token_ids = shared_u32_buf(&ctx, &plan.token_ids_sorted);
        let routes = shared_f32_buf(&ctx, &plan.route_weights_sorted);
        let expert_offsets = shared_u32_buf(&ctx, &plan.expert_offsets);
        let expert_counts = shared_u32_buf(&ctx, &plan.expert_counts);
        let id_block_experts = shared_u32_buf(&ctx, &plan.id_block_experts);
        let id_block_local0 = shared_u32_buf(&ctx, &plan.id_block_local0);
        let expert_rank_offsets = shared_u32_buf(&ctx, &plan.expert_rank_offsets);
        let expert_rank_counts = shared_u32_buf(&ctx, &plan.expert_rank_counts);
        let rank_block_experts = shared_u32_buf(&ctx, &plan.rank_block_experts);
        let rank_block_local0 = shared_u32_buf(&ctx, &plan.rank_block_local0);

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        qwen_moe_prefill_mulmmid_v4_encode(
            &ctx,
            &enc,
            &carrier,
            &accum,
            &accum.norm_all_dev,
            &gate_raw,
            0,
            &up_raw,
            0,
            &down_q4_raw,
            0,
            q4_stride,
            q4_stride,
            down_q4_stride,
            &token_ids,
            &routes,
            &expert_offsets,
            &expert_counts,
            &id_block_experts,
            &id_block_local0,
            id_block_count,
            &expert_rank_offsets,
            &expert_rank_counts,
            &rank_block_experts,
            &rank_block_local0,
            &plan.rank_block_offsets,
            &plan.rank_block_counts,
            false,
        )
        .expect("encode v4 q4 chain");
        qwen_moe_prefill_mulmmid_v4_encode(
            &ctx,
            &enc,
            &carrier,
            &accum,
            &accum.norm_all_dev,
            &gate_raw,
            0,
            &up_raw,
            0,
            &down_q6_raw,
            0,
            q4_stride,
            q4_stride,
            down_q6_stride,
            &token_ids,
            &routes,
            &expert_offsets,
            &expert_counts,
            &id_block_experts,
            &id_block_local0,
            id_block_count,
            &expert_rank_offsets,
            &expert_rank_counts,
            &rank_block_experts,
            &rank_block_local0,
            &plan.rank_block_offsets,
            &plan.rank_block_counts,
            true,
        )
        .expect("encode v4 q6 chain");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v4_chain_encoder_rejects_rank_metadata_mismatch() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _v4_env = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4", "1");
        let ctx = build_metal_context().expect("no metal device");
        let hidden_dim = 256usize;
        let ffn_dim = 256usize;
        let seq_len = 1usize;
        let n_expert_used = 1usize;
        let n_expert = 1usize;
        let plan = build_qwen_moe_mul_mm_id_v4_plan_token_major(
            &[0],
            &[0],
            &[1.0],
            n_expert,
            seq_len,
            n_expert_used,
            64,
        )
        .expect("v4 plan");
        let id_block_count = plan.id_block_experts.len();
        let q4_stride = ffn_dim * (hidden_dim / 256) * 144;
        let gate_raw = shared_u8_buf(&ctx, &vec![0u8; q4_stride * n_expert]);
        let up_raw = shared_u8_buf(&ctx, &vec![0u8; q4_stride * n_expert]);
        let down_raw = shared_u8_buf(&ctx, &vec![0u8; hidden_dim * (ffn_dim / 256) * 144]);
        let accum = QwenMoePrefillAccumCarrier::new(&ctx, hidden_dim, seq_len);
        accum.upload_norm_all(&vec![0.0f32; hidden_dim * seq_len]);
        let carrier = QwenMoeMulMmIdV4Carrier::new(
            &ctx,
            seq_len,
            n_expert_used,
            hidden_dim,
            ffn_dim,
            n_expert,
            id_block_count,
            64 * 1024 * 1024,
        )
        .expect("v4 carrier");
        let token_ids = shared_u32_buf(&ctx, &plan.token_ids_sorted);
        let routes = shared_f32_buf(&ctx, &plan.route_weights_sorted);
        let expert_offsets = shared_u32_buf(&ctx, &plan.expert_offsets);
        let expert_counts = shared_u32_buf(&ctx, &plan.expert_counts);
        let id_block_experts = shared_u32_buf(&ctx, &plan.id_block_experts);
        let id_block_local0 = shared_u32_buf(&ctx, &plan.id_block_local0);
        let expert_rank_offsets = shared_u32_buf(&ctx, &plan.expert_rank_offsets);
        let expert_rank_counts = shared_u32_buf(&ctx, &plan.expert_rank_counts);
        let rank_block_experts = shared_u32_buf(&ctx, &plan.rank_block_experts);
        let rank_block_local0 = shared_u32_buf(&ctx, &plan.rank_block_local0);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();

        let err = qwen_moe_prefill_mulmmid_v4_encode(
            &ctx,
            &enc,
            &carrier,
            &accum,
            &accum.norm_all_dev,
            &gate_raw,
            0,
            &up_raw,
            0,
            &down_raw,
            0,
            q4_stride,
            q4_stride,
            hidden_dim * (ffn_dim / 256) * 144,
            &token_ids,
            &routes,
            &expert_offsets,
            &expert_counts,
            &id_block_experts,
            &id_block_local0,
            id_block_count,
            &expert_rank_offsets,
            &expert_rank_counts,
            &rank_block_experts,
            &rank_block_local0,
            &[],
            &[],
            false,
        )
        .expect_err("rank metadata mismatch should be rejected");
        assert_eq!(err, QwenMoeIdPreflightReason::RankCountMismatch);
        enc.endEncoding();
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_v4_carrier_allocates_only_gate_and_up_scratch() {
        let ctx = build_metal_context().expect("no metal device");
        let carrier = QwenMoeMulMmIdV4Carrier::new(&ctx, 4, 2, 512, 256, 3, 4, 64 * 1024 * 1024)
            .expect("carrier");
        assert_eq!(carrier.down_scratch_bytes(), 0);
        assert_eq!(
            carrier.scratch_bytes(),
            carrier.gate_bytes() + carrier.up_bytes() + carrier.metadata_bytes()
        );
    }
    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_map0_matches_cpu() {
        let ctx = build_metal_context().expect("no metal device");
        let selected_experts = [2u32, 0, 3, 1];
        let (want_tpe, want_ids) =
            qwen_moe_llama_map0_cpu(&selected_experts, 2, 2, 4).expect("CPU map");
        let selected_buf = shared_u32_buf(&ctx, &selected_experts);
        let tpe_buf = shared_u32_buf(&ctx, &[u32::MAX; 4]);
        let ids_buf = shared_u32_buf(&ctx, &[i32::MIN as u32; 8]);

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        qwen_moe_llama_id_map0_encode(&ctx, &enc, &selected_buf, &tpe_buf, &ids_buf, 2, 2, 4)
            .expect("encode map0");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let got_tpe =
            unsafe { std::slice::from_raw_parts(tpe_buf.contents().as_ptr() as *const u32, 4) };
        let got_ids =
            unsafe { std::slice::from_raw_parts(ids_buf.contents().as_ptr() as *const i32, 8) };
        assert_eq!(got_tpe, want_tpe.as_slice());
        for expert in 0..4 {
            let count = want_tpe[expert] as usize;
            let base = expert * 2;
            assert_eq!(&got_ids[base..base + count], &want_ids[base..base + count]);
        }
    }

    fn qwen_moe_llama_test_block(quant: QwenMoeLlamaIdQuant, seed: usize) -> Vec<u8> {
        let d = half::f16::from_f32(0.006 + (seed % 11) as f32 * 0.0005).to_le_bytes();
        match quant {
            QwenMoeLlamaIdQuant::Q4K => {
                let mut block = crate::tests_fixture::q4k_block_fixed();
                block[0..2].copy_from_slice(&d);
                block[2..4].copy_from_slice(
                    &half::f16::from_f32(0.003 + (seed % 7) as f32 * 0.00025).to_le_bytes(),
                );
                block
            }
            QwenMoeLlamaIdQuant::Q5K => {
                let mut block = crate::tests_fixture::q5k_block_fixed();
                block[0..2].copy_from_slice(&d);
                block[2..4].copy_from_slice(
                    &half::f16::from_f32(0.002 + (seed % 5) as f32 * 0.00025).to_le_bytes(),
                );
                block
            }
            QwenMoeLlamaIdQuant::Q6K => {
                let mut block = crate::tests_fixture::q6k_block_fixed();
                block[208..210].copy_from_slice(&d);
                block
            }
            QwenMoeLlamaIdQuant::Q8Zero => unreachable!(),
        }
    }

    fn qwen_moe_llama_test_dequant(quant: QwenMoeLlamaIdQuant, block: &[u8]) -> [f32; 256] {
        match quant {
            QwenMoeLlamaIdQuant::Q4K => crate::tests_fixture::q4k_dequant(block),
            QwenMoeLlamaIdQuant::Q5K => crate::tests_fixture::q5k_dequant(block),
            QwenMoeLlamaIdQuant::Q6K => crate::tests_fixture::q6k_dequant(block),
            QwenMoeLlamaIdQuant::Q8Zero => unreachable!(),
        }
    }

    fn qwen_moe_llama_test_input(n_tokens: usize, input_dim: usize) -> Vec<f32> {
        (0..n_tokens * input_dim)
            .map(|index| ((index * 17 + 9) % 97) as f32 * 0.002 - 0.09)
            .collect()
    }

    fn qwen_moe_llama_assert_close(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = 0usize;
        for index in 0..got.len() {
            assert!(
                got[index].is_finite(),
                "{label}: candidate became non-finite at {index}: {}",
                got[index],
            );
            assert!(
                want[index].is_finite(),
                "{label}: baseline became non-finite at {index}: {}",
                want[index],
            );
            let abs = (got[index] - want[index]).abs();
            let rel = abs / want[index].abs().max(1.0);
            if abs > max_abs {
                max_abs = abs;
                worst = index;
            }
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_abs <= 0.025 && max_rel <= 0.004,
            "{label}: max_abs={max_abs:e} max_rel={max_rel:e} worst={worst} got={} want={}",
            got[worst],
            want[worst],
        );
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_llama_mul_mm_id_matches_cpu_oracle() {
        let ctx = build_metal_context().expect("no metal device");
        if !ctx.tensorops_capable {
            eprintln!("skipping: Metal tensor operations unavailable");
            return;
        }

        const INPUT_DIM: usize = 256;
        const N_EXPERT: usize = 4;
        const N_EXPERT_USED: usize = 2;
        for quant in [
            QwenMoeLlamaIdQuant::Q4K,
            QwenMoeLlamaIdQuant::Q5K,
            QwenMoeLlamaIdQuant::Q6K,
        ] {
            let block_bytes = match quant {
                QwenMoeLlamaIdQuant::Q4K => 144,
                QwenMoeLlamaIdQuant::Q5K => 176,
                QwenMoeLlamaIdQuant::Q6K => 210,
                QwenMoeLlamaIdQuant::Q8Zero => unreachable!(),
            };
            for input_type in [QwenMoeLlamaIdInput::F32, QwenMoeLlamaIdInput::F16] {
                for n_tokens in [31usize, 32, 33] {
                    let selected = (0..n_tokens)
                        .flat_map(|token| {
                            if token % 2 == 0 {
                                [0u32, 2u32]
                            } else {
                                [2u32, 0u32]
                            }
                        })
                        .collect::<Vec<_>>();
                    let (tpe, ids) =
                        qwen_moe_llama_map0_cpu(&selected, n_tokens, N_EXPERT_USED, N_EXPERT)
                            .expect("valid test routes");
                    assert_eq!(tpe[1], 0);
                    assert_eq!(tpe[3], 0);
                    let tpe_buf = shared_u32_buf(&ctx, &tpe);
                    let ids_bits = ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
                    let ids_buf = shared_u32_buf(&ctx, &ids_bits);
                    let input_f32 = qwen_moe_llama_test_input(n_tokens, INPUT_DIM);
                    let (input_buf, oracle_input) = match input_type {
                        QwenMoeLlamaIdInput::F32 => (
                            shared_f32_buf(&ctx, &input_f32),
                            input_f32
                                .iter()
                                .map(|&value| half::f16::from_f32(value).to_f32())
                                .collect::<Vec<_>>(),
                        ),
                        QwenMoeLlamaIdInput::F16 => {
                            let (bytes, rounded) = half_bytes_and_cpu(&input_f32);
                            (shared_u8_buf(&ctx, &bytes), rounded)
                        }
                    };

                    for output_dim in [63usize, 64, 65] {
                        let expert_weight_bytes = output_dim * block_bytes;
                        let mut weights = Vec::with_capacity(N_EXPERT * expert_weight_bytes);
                        let mut dequantized = Vec::with_capacity(N_EXPERT * output_dim * INPUT_DIM);
                        for expert in 0..N_EXPERT {
                            for row in 0..output_dim {
                                let block = qwen_moe_llama_test_block(quant, expert * 131 + row);
                                dequantized.extend(
                                    qwen_moe_llama_test_dequant(quant, &block)
                                        .into_iter()
                                        .map(|value| half::f16::from_f32(value).to_f32()),
                                );
                                weights.extend_from_slice(&block);
                            }
                        }
                        let weight_offset = std::mem::align_of::<u16>();
                        let mut weight_storage = Vec::with_capacity(weight_offset + weights.len());
                        weight_storage.resize(weight_offset, 0);
                        weight_storage.extend_from_slice(&weights);
                        let weights_buf = shared_u8_buf(&ctx, &weight_storage);
                        let output_len = n_tokens * N_EXPERT_USED * output_dim;
                        let output_buf = empty_f32_buf(&ctx, output_len);

                        let command = ctx.queue.commandBuffer().expect("command buffer");
                        let encoder = command.computeCommandEncoder().expect("compute encoder");
                        qwen_moe_llama_mul_mm_id_encode(
                            &ctx,
                            &encoder,
                            quant,
                            input_type,
                            &weights_buf,
                            weight_offset,
                            &input_buf,
                            &tpe_buf,
                            &ids_buf,
                            &output_buf,
                            QwenMoeLlamaIdMatmulShape {
                                input_dim: INPUT_DIM,
                                output_dim,
                                n_tokens,
                                n_expert: N_EXPERT,
                                n_expert_used: N_EXPERT_USED,
                                expert_weight_bytes,
                            },
                        )
                        .expect("ID matmul encode");
                        encoder.endEncoding();
                        command.commit();
                        command.waitUntilCompleted();

                        let mut oracle = vec![0.0f32; output_len];
                        for token in 0..n_tokens {
                            let input = &oracle_input[token * INPUT_DIM..(token + 1) * INPUT_DIM];
                            for rank in 0..N_EXPERT_USED {
                                let expert = selected[token * N_EXPERT_USED + rank] as usize;
                                for row in 0..output_dim {
                                    let weight_base = (expert * output_dim + row) * INPUT_DIM;
                                    oracle[(token * N_EXPERT_USED + rank) * output_dim + row] =
                                        dequantized[weight_base..weight_base + INPUT_DIM]
                                            .iter()
                                            .zip(input)
                                            .map(|(&weight, &activation)| weight * activation)
                                            .sum();
                                }
                            }
                        }
                        let got = readback(&output_buf, output_len);
                        qwen_moe_llama_assert_close(
                            &format!(
                                "{quant:?}/{input_type:?}/tokens={n_tokens}/rows={output_dim}"
                            ),
                            &got,
                            &oracle,
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_llama_mul_mm_id_shared_q8_0_f32_matches_cpu_oracle() {
        let ctx = build_metal_context().expect("no metal device");
        if !ctx.tensorops_capable {
            eprintln!("skipping: Metal tensor operations unavailable");
            return;
        }

        const INPUT_DIM: usize = 256;
        for n_tokens in [31usize, 32, 33] {
            let input = qwen_moe_llama_test_input(n_tokens, INPUT_DIM);
            let oracle_input = input
                .iter()
                .map(|&value| half::f16::from_f32(value).to_f32())
                .collect::<Vec<_>>();
            let input_buf = shared_f32_buf(&ctx, &input);
            for output_dim in [63usize, 64, 65] {
                let mut weights = Vec::with_capacity(output_dim * (INPUT_DIM / 32) * 34);
                let mut dequantized = Vec::with_capacity(output_dim * INPUT_DIM);
                for row in 0..output_dim {
                    for block_index in 0..INPUT_DIM / 32 {
                        let seed = row * 17 + block_index;
                        let mut block = vec![0u8; 34];
                        block[0..2].copy_from_slice(
                            &half::f16::from_f32(0.008 + (seed % 9) as f32 * 0.0005).to_le_bytes(),
                        );
                        for (index, quant) in block[2..].iter_mut().enumerate() {
                            *quant = (((seed * 13 + index * 7) % 127) as i8 - 63) as u8;
                        }
                        dequantized.extend(
                            crate::tests_fixture::q8_0_dequant(&block)
                                .into_iter()
                                .map(|value| half::f16::from_f32(value).to_f32()),
                        );
                        weights.extend_from_slice(&block);
                    }
                }
                let weights_buf = shared_u8_buf(&ctx, &weights);
                let output_len = n_tokens * output_dim;
                let output_buf = empty_f32_buf(&ctx, output_len);
                let command = ctx.queue.commandBuffer().expect("command buffer");
                let encoder = command.computeCommandEncoder().expect("compute encoder");
                qwen_moe_shared_mul_mm_q8_0_f32_encode(
                    &ctx,
                    &encoder,
                    &weights_buf,
                    0,
                    &input_buf,
                    &output_buf,
                    INPUT_DIM,
                    output_dim,
                    n_tokens,
                )
                .expect("shared Q8_0 encode");
                encoder.endEncoding();
                command.commit();
                command.waitUntilCompleted();

                let mut oracle = vec![0.0f32; output_len];
                for token in 0..n_tokens {
                    let activation = &oracle_input[token * INPUT_DIM..(token + 1) * INPUT_DIM];
                    for row in 0..output_dim {
                        oracle[token * output_dim + row] = dequantized
                            [row * INPUT_DIM..(row + 1) * INPUT_DIM]
                            .iter()
                            .zip(activation)
                            .map(|(&weight, &input)| weight * input)
                            .sum();
                    }
                }
                let got = readback(&output_buf, output_len);
                qwen_moe_llama_assert_close(
                    &format!("Q8_0/F32/tokens={n_tokens}/rows={output_dim}"),
                    &got,
                    &oracle,
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_llama_exact_q8k_q4_q6_match_cpu_bits() {
        use rnb_cpu::gemm::activation_q8::quantize_input_q8k;
        use rnb_cpu::gemm::neon_dot::{dot_q4_k_q8k_neon_ggml_align, dot_q6_k_q8k_neon_ggml_align};

        let ctx =
            crate::compute::build_metal_context_with_opts(false, true).expect("no metal device");
        if !ctx.tensorops_capable {
            eprintln!("skipping: Metal tensor operations unavailable");
            return;
        }
        const N_TOKENS: usize = 3;
        const INPUT_DIM: usize = 512;
        let mut input = vec![0.0f32; N_TOKENS * INPUT_DIM];
        for token in 0..N_TOKENS {
            for block in 0..INPUT_DIM / 256 {
                let base = token * INPUT_DIM + block * 256;
                for i in 0..256 {
                    input[base + i] =
                        (((token * 97 + block * 53 + i * 29) % 251) as f32 - 125.0) / 31.0;
                }
                if (token + block) % 2 == 0 {
                    input[base] = 127.0;
                    input[base + 1] = -127.0;
                } else {
                    input[base] = -127.0;
                    input[base + 1] = 127.0;
                }
                input[base + 2..base + 8].copy_from_slice(&[-0.5, -1.5, -2.5, 0.5, 1.5, 2.5]);
            }
        }
        let cpu_q8 = quantize_input_q8k(&input);
        let input_dev = shared_f32_buf(&ctx, &input);
        let q8_bytes = cpu_q8.len() * QWEN_MOE_LLAMA_Q8K_BLOCK_BYTES;
        let q8_dev = ctx
            .device
            .newBufferWithLength_options(q8_bytes, MTLResourceOptions::StorageModeShared)
            .expect("q8 buffer");

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = crate::compute::chain_compute_encoder(&ctx, &command);
        qwen_moe_llama_quantize_q8k_f32_encode(&ctx, &encoder, &input_dev, &q8_dev, input.len())
            .expect("q8 quantize");
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let cpu_q8_bytes =
            unsafe { std::slice::from_raw_parts(cpu_q8.as_ptr() as *const u8, q8_bytes) };
        let gpu_q8_bytes = unsafe {
            std::slice::from_raw_parts(q8_dev.contents().as_ptr() as *const u8, q8_bytes)
        };
        assert_eq!(gpu_q8_bytes, cpu_q8_bytes, "GPU Q8_K bytes differ from CPU");

        let tpe = shared_u32_buf(&ctx, &[N_TOKENS as u32]);
        let ids = shared_u32_buf(&ctx, &[0, 1, 2]);
        for output_dim in [31usize, 32, 33] {
            for quant in [QwenMoeLlamaIdQuant::Q4K, QwenMoeLlamaIdQuant::Q6K] {
                let (weight_bytes, _) =
                    qwen_moe_llama_test_sparse_matrix(quant, 1, output_dim, INPUT_DIM, 991);
                let weights = shared_u8_buf(&ctx, &weight_bytes);
                let output = empty_f32_buf(&ctx, N_TOKENS * output_dim);
                let command = ctx.queue.commandBuffer().expect("command buffer");
                let encoder = crate::compute::chain_compute_encoder(&ctx, &command);
                qwen_moe_llama_mul_mm_id_q8k_encode(
                    &ctx,
                    &encoder,
                    quant,
                    &weights,
                    0,
                    &q8_dev,
                    &tpe,
                    &ids,
                    &output,
                    QwenMoeLlamaIdMatmulShape {
                        input_dim: INPUT_DIM,
                        output_dim,
                        n_tokens: N_TOKENS,
                        n_expert: 1,
                        n_expert_used: 1,
                        expert_weight_bytes: weight_bytes.len(),
                    },
                )
                .expect("exact sparse q8k");
                encoder.endEncoding();
                command.commit();
                command.waitUntilCompleted();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

                let blocks_per_row = INPUT_DIM / 256;
                let bytes_per_row = blocks_per_row
                    * match quant {
                        QwenMoeLlamaIdQuant::Q4K => 144,
                        QwenMoeLlamaIdQuant::Q6K => 210,
                        _ => unreachable!(),
                    };
                let mut want = vec![0.0f32; N_TOKENS * output_dim];
                for token in 0..N_TOKENS {
                    let q8 = &cpu_q8[token * blocks_per_row..(token + 1) * blocks_per_row];
                    for row in 0..output_dim {
                        let row_bytes =
                            &weight_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                        want[token * output_dim + row] = unsafe {
                            match quant {
                                QwenMoeLlamaIdQuant::Q4K => {
                                    dot_q4_k_q8k_neon_ggml_align(row_bytes, q8, blocks_per_row)
                                }
                                QwenMoeLlamaIdQuant::Q6K => {
                                    dot_q6_k_q8k_neon_ggml_align(row_bytes, q8, blocks_per_row)
                                }
                                _ => unreachable!(),
                            }
                        };
                    }
                }
                let got = readback(&output, want.len());
                assert_eq!(
                    got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    "{quant:?} rows={output_dim}"
                );
            }
        }
    }

    fn qwen_moe_llama_test_sparse_matrix(
        quant: QwenMoeLlamaIdQuant,
        n_expert: usize,
        output_dim: usize,
        input_dim: usize,
        seed_base: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(input_dim % 256, 0);
        let mut weights = Vec::new();
        let mut dequantized = Vec::with_capacity(n_expert * output_dim * input_dim);
        for expert in 0..n_expert {
            for row in 0..output_dim {
                for block_index in 0..input_dim / 256 {
                    let seed = seed_base + expert * 1009 + row * 17 + block_index * 131;
                    let block = qwen_moe_llama_test_block(quant, seed);
                    dequantized.extend(
                        qwen_moe_llama_test_dequant(quant, &block)
                            .into_iter()
                            .map(|value| half::f16::from_f32(value).to_f32()),
                    );
                    weights.extend_from_slice(&block);
                }
            }
        }
        (weights, dequantized)
    }

    fn qwen_moe_llama_test_q8_matrix(
        output_dim: usize,
        input_dim: usize,
        seed_base: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(input_dim % 32, 0);
        let mut weights = Vec::with_capacity(output_dim * (input_dim / 32) * 34);
        let mut dequantized = Vec::with_capacity(output_dim * input_dim);
        for row in 0..output_dim {
            for block_index in 0..input_dim / 32 {
                let seed = seed_base + row * 29 + block_index * 7;
                let mut block = vec![0u8; 34];
                block[0..2].copy_from_slice(
                    &half::f16::from_f32(0.004 + (seed % 13) as f32 * 0.00025).to_le_bytes(),
                );
                for (index, quantized) in block[2..].iter_mut().enumerate() {
                    *quantized = (((seed * 11 + index * 5) % 63) as i8 - 31) as u8;
                }
                dequantized.extend(
                    crate::tests_fixture::q8_0_dequant(&block)
                        .into_iter()
                        .map(|value| half::f16::from_f32(value).to_f32()),
                );
                weights.extend_from_slice(&block);
            }
        }
        (weights, dequantized)
    }

    fn qwen_moe_llama_test_shared_matrix(
        quant: QwenMoeLlamaIdQuant,
        output_dim: usize,
        input_dim: usize,
        seed_base: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        match quant {
            QwenMoeLlamaIdQuant::Q8Zero => {
                qwen_moe_llama_test_q8_matrix(output_dim, input_dim, seed_base)
            }
            QwenMoeLlamaIdQuant::Q4K | QwenMoeLlamaIdQuant::Q6K => {
                qwen_moe_llama_test_sparse_matrix(quant, 1, output_dim, input_dim, seed_base)
            }
            QwenMoeLlamaIdQuant::Q5K => panic!("shared Q5_K is unsupported"),
        }
    }

    fn qwen_moe_llama_test_matmul(
        weights: &[f32],
        input: &[f32],
        output_dim: usize,
        input_dim: usize,
    ) -> Vec<f32> {
        assert_eq!(weights.len(), output_dim * input_dim);
        assert_eq!(input.len(), input_dim);
        let rounded_input = input
            .iter()
            .map(|&value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        (0..output_dim)
            .map(|row| {
                weights[row * input_dim..(row + 1) * input_dim]
                    .iter()
                    .zip(&rounded_input)
                    .map(|(&weight, &activation)| weight * activation)
                    .sum()
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen_moe_llama_prefill_cpu_oracle(
        norm_all: &[f32],
        selected_experts: &[u32],
        route_weights: &[f32],
        shared_route_weights: &[f32],
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        shared_gate: &[f32],
        shared_up: &[f32],
        shared_down: &[f32],
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; n_tokens * hidden_dim];
        for token in 0..n_tokens {
            let norm = &norm_all[token * hidden_dim..(token + 1) * hidden_dim];
            for rank in 0..n_expert_used {
                let slot = token * n_expert_used + rank;
                let expert = selected_experts[slot] as usize;
                assert!(expert < n_expert);
                let gate_base = expert * ffn_dim * hidden_dim;
                let down_base = expert * hidden_dim * ffn_dim;
                let gate_value = qwen_moe_llama_test_matmul(
                    &gate[gate_base..gate_base + ffn_dim * hidden_dim],
                    norm,
                    ffn_dim,
                    hidden_dim,
                );
                let up_value = qwen_moe_llama_test_matmul(
                    &up[gate_base..gate_base + ffn_dim * hidden_dim],
                    norm,
                    ffn_dim,
                    hidden_dim,
                );
                let activation = gate_value
                    .iter()
                    .zip(&up_value)
                    .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
                    .collect::<Vec<_>>();
                let down_value = qwen_moe_llama_test_matmul(
                    &down[down_base..down_base + hidden_dim * ffn_dim],
                    &activation,
                    hidden_dim,
                    ffn_dim,
                );
                for row in 0..hidden_dim {
                    let weighted = route_weights[slot] * down_value[row];
                    let out = &mut output[token * hidden_dim + row];
                    if rank == 0 {
                        *out = weighted;
                    } else {
                        *out += weighted;
                    }
                }
            }

            let shared_gate_value =
                qwen_moe_llama_test_matmul(shared_gate, norm, ffn_dim, hidden_dim);
            let shared_up_value = qwen_moe_llama_test_matmul(shared_up, norm, ffn_dim, hidden_dim);
            let shared_activation = shared_gate_value
                .iter()
                .zip(&shared_up_value)
                .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
                .collect::<Vec<_>>();
            let shared_down_value =
                qwen_moe_llama_test_matmul(shared_down, &shared_activation, hidden_dim, ffn_dim);
            for row in 0..hidden_dim {
                output[token * hidden_dim + row] +=
                    shared_route_weights[token] * shared_down_value[row];
            }
        }
        output
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_llama_prefill_whole_layer_matches_cpu_and_reuse_is_exact() {
        let ctx =
            crate::compute::build_metal_context_with_opts(false, true).expect("no metal device");
        if !ctx.tensorops_capable {
            eprintln!("skipping: Metal tensor operations unavailable");
            return;
        }

        const N_TOKENS: usize = 2;
        const N_EXPERT: usize = 6;
        const N_EXPERT_USED: usize = 3;
        const HIDDEN_DIM: usize = 256;
        const FFN_DIM: usize = 256;
        let selected_experts = [0, 2, 4, 3, 0, 2];
        let route_weights = [0.55, 0.30, 0.15, 0.20, 0.70, 0.10];
        let shared_route_weights = [0.35, 0.80];
        let norm_all = qwen_moe_llama_test_input(N_TOKENS, HIDDEN_DIM);
        let norm_all_dev = shared_f32_buf(&ctx, &norm_all);
        let q4_shared = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q4K,
        };
        let q6_shared = QwenMoeLlamaIdQuantSet {
            down: QwenMoeLlamaIdQuant::Q6K,
            ..q4_shared
        };
        let q8_shared = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        };

        for (sparse_down_quant, shared_quant) in [
            (QwenMoeLlamaIdQuant::Q4K, q4_shared),
            (QwenMoeLlamaIdQuant::Q6K, q6_shared),
            (QwenMoeLlamaIdQuant::Q5K, q8_shared),
            (QwenMoeLlamaIdQuant::Q6K, q8_shared),
        ] {
            let (shared_gate_bytes, shared_gate_cpu) =
                qwen_moe_llama_test_shared_matrix(shared_quant.gate, FFN_DIM, HIDDEN_DIM, 7001);
            let (shared_up_bytes, shared_up_cpu) =
                qwen_moe_llama_test_shared_matrix(shared_quant.up, FFN_DIM, HIDDEN_DIM, 8009);
            let (shared_down_bytes, shared_down_cpu) =
                qwen_moe_llama_test_shared_matrix(shared_quant.down, HIDDEN_DIM, FFN_DIM, 9001);
            let shared_gate_dev = shared_u8_buf(&ctx, &shared_gate_bytes);
            let shared_up_dev = shared_u8_buf(&ctx, &shared_up_bytes);
            let shared_down_dev = shared_u8_buf(&ctx, &shared_down_bytes);
            let sparse_quant = QwenMoeLlamaIdQuantSet {
                gate: QwenMoeLlamaIdQuant::Q4K,
                up: QwenMoeLlamaIdQuant::Q4K,
                down: sparse_down_quant,
            };
            let (gate_bytes, gate_cpu) = qwen_moe_llama_test_sparse_matrix(
                sparse_quant.gate,
                N_EXPERT,
                FFN_DIM,
                HIDDEN_DIM,
                101,
            );
            let (up_bytes, up_cpu) = qwen_moe_llama_test_sparse_matrix(
                sparse_quant.up,
                N_EXPERT,
                FFN_DIM,
                HIDDEN_DIM,
                2003,
            );
            let (down_bytes, down_cpu) = qwen_moe_llama_test_sparse_matrix(
                sparse_quant.down,
                N_EXPERT,
                HIDDEN_DIM,
                FFN_DIM,
                4001,
            );
            let gate_dev = shared_u8_buf(&ctx, &gate_bytes);
            let up_dev = shared_u8_buf(&ctx, &up_bytes);
            let down_dev = shared_u8_buf(&ctx, &down_bytes);
            let QwenMoeLlamaIdPreflight::Run(plan) = qwen_moe_llama_id_preflight(
                true,
                true,
                N_TOKENS,
                N_EXPERT,
                N_EXPERT_USED,
                HIDDEN_DIM,
                FFN_DIM,
                usize::MAX,
            )
            .expect("valid whole-layer plan") else {
                panic!("whole-layer shape must run");
            };
            let carrier = QwenMoeLlamaIdCarrier::new(
                &ctx,
                plan,
                &selected_experts,
                &route_weights,
                &shared_route_weights,
            )
            .expect("whole-layer carrier");
            assert_eq!(carrier.allocation_bytes(), plan.total_bytes);
            assert_eq!(carrier.gate_dev.length(), plan.gate_bytes);
            assert_eq!(carrier.up_dev.length(), plan.up_bytes);
            assert_eq!(carrier.down_rank_dev.length(), plan.down_rank_bytes);
            assert_eq!(
                carrier.sparse_moe_out_dev.length(),
                plan.sparse_moe_out_bytes
            );
            assert_eq!(carrier.shared_out_dev.length(), plan.shared_out_bytes);

            let oracle = qwen_moe_llama_prefill_cpu_oracle(
                &norm_all,
                &selected_experts,
                &route_weights,
                &shared_route_weights,
                N_TOKENS,
                N_EXPERT,
                N_EXPERT_USED,
                HIDDEN_DIM,
                FFN_DIM,
                &gate_cpu,
                &up_cpu,
                &down_cpu,
                &shared_gate_cpu,
                &shared_up_cpu,
                &shared_down_cpu,
            );
            let mut expected_bits = None;
            for iteration in 0..20 {
                for map_reuse in [false, true] {
                    let command = ctx.queue.commandBuffer().expect("command buffer");
                    let encoder = crate::compute::chain_compute_encoder(&ctx, &command);
                    let trace = qwen_moe_llama_prefill_encode(
                        &ctx,
                        &encoder,
                        &carrier,
                        &norm_all_dev,
                        &gate_dev,
                        0,
                        &up_dev,
                        0,
                        &down_dev,
                        0,
                        sparse_quant,
                        &shared_gate_dev,
                        0,
                        &shared_up_dev,
                        0,
                        &shared_down_dev,
                        0,
                        shared_quant,
                        map_reuse,
                        None,
                    )
                    .expect("whole-layer encode");
                    encoder.endEncoding();
                    command.commit();
                    command.waitUntilCompleted();

                    assert_eq!(trace.map_dispatches, if map_reuse { 1 } else { 3 });
                    assert_eq!(trace.gate_dispatches, 2);
                    assert_eq!(trace.up_dispatches, 2);
                    assert_eq!(trace.activation_dispatches, 2);
                    assert_eq!(trace.down_dispatches, 2);
                    assert_eq!(trace.reduce_dispatches, 2);
                    assert_eq!(trace.scratch_bytes, plan.total_bytes);
                    let output = readback(&carrier.sparse_moe_out_dev, N_TOKENS * HIDDEN_DIM);
                    qwen_moe_llama_assert_close(
                        &format!(
                            "whole-layer/sparse={sparse_down_quant:?}/shared={shared_quant:?}/iteration={iteration}/reuse={map_reuse}"
                        ),
                        &output,
                        &oracle,
                    );
                    let output_bits = output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>();
                    if let Some(expected_bits) = &expected_bits {
                        assert_eq!(
                            &output_bits, expected_bits,
                            "concurrent execution or map reuse changed final output for \
                             sparse={sparse_down_quant:?}, shared={shared_quant:?} at \
                             iteration {iteration}, reuse={map_reuse}"
                        );
                    } else {
                        expected_bits = Some(output_bits);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_prefill_rank_reduce_is_left_fold_before_shared_add() {
        let ctx = build_metal_context().expect("no metal device");
        let slot_values = shared_f32_buf(&ctx, &[1.0e20, 3.25, -1.0e20]);
        let route_weights = shared_f32_buf(&ctx, &[1.0, 1.0, 1.0]);
        let sparse_out = empty_f32_buf(&ctx, 1);
        let shared_out = shared_f32_buf(&ctx, &[2.75]);

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("compute encoder");
        qwen_moe_llama_weighted_rank_reduce_f32_encode(
            &ctx,
            &encoder,
            &slot_values,
            &route_weights,
            &sparse_out,
            1,
            3,
            1,
        )
        .expect("ordered rank reduction");
        encode_residual_add_inline_dim(&ctx, &encoder, &sparse_out, &shared_out, 1)
            .expect("sparse then shared add");
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        let mut expected = 1.0e20f32;
        expected += 3.25;
        expected += -1.0e20;
        expected += 2.75;
        let got = readback(&sparse_out, 1)[0];
        assert_eq!(got.to_bits(), expected.to_bits());
        assert_ne!(
            got.to_bits(),
            ((1.0e20f32 + 3.25) + (-1.0e20f32 + 2.75)).to_bits(),
            "tree/shared-first grouping unexpectedly matched the required left fold"
        );
    }

    #[test]
    #[ignore = "requires a TensorOps-capable Metal device"]
    fn qwen_moe_llama_id_matches_legacy_selected_base_stage_oracle() {
        #[derive(Default)]
        struct LegacySparseStages {
            norm: Vec<f32>,
            gate: Vec<f32>,
            up: Vec<f32>,
            activation: Vec<f32>,
            down: Vec<f32>,
            output: Vec<f32>,
        }

        #[derive(Debug)]
        struct StageDiff {
            first_bits: Option<(usize, u32, u32)>,
            max_abs: f32,
            max_rel: f32,
        }

        fn submit(
            ctx: &MetalContext,
            encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
        ) {
            let command = ctx.queue.commandBuffer().expect("command buffer");
            let encoder = command.computeCommandEncoder().expect("compute encoder");
            encode(&encoder);
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
        }

        fn readback_u32(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<u32> {
            unsafe {
                std::slice::from_raw_parts(buf.contents().as_ptr() as *const u32, len).to_vec()
            }
        }

        fn readback_f16(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<f32> {
            unsafe {
                std::slice::from_raw_parts(buf.contents().as_ptr() as *const u16, len)
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                    .collect()
            }
        }

        fn write_f32(buf: &ProtocolObject<dyn MTLBuffer>, values: &[f32]) {
            assert!(buf.length() >= std::mem::size_of_val(values));
            unsafe {
                std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut f32, values.len())
                    .copy_from_slice(values);
            }
        }

        fn align_group(aligned: &mut [f32], grouped: &[f32], slot_indices: &[usize], width: usize) {
            assert_eq!(grouped.len(), slot_indices.len() * width);
            for (local, &slot) in slot_indices.iter().enumerate() {
                aligned[slot * width..(slot + 1) * width]
                    .copy_from_slice(&grouped[local * width..(local + 1) * width]);
            }
        }

        fn stage_diff(lhs: &[f32], rhs: &[f32]) -> StageDiff {
            assert_eq!(lhs.len(), rhs.len());
            let mut diff = StageDiff {
                first_bits: None,
                max_abs: 0.0,
                max_rel: 0.0,
            };
            for (index, (&left, &right)) in lhs.iter().zip(rhs).enumerate() {
                assert!(
                    left.is_finite() && right.is_finite(),
                    "non-finite stage value at {index}: {left} vs {right}"
                );
                if diff.first_bits.is_none() && left.to_bits() != right.to_bits() {
                    diff.first_bits = Some((index, left.to_bits(), right.to_bits()));
                }
                let abs = (left - right).abs();
                let rel = abs / left.abs().max(right.abs()).max(1.0e-6);
                diff.max_abs = diff.max_abs.max(abs);
                diff.max_rel = diff.max_rel.max(rel);
            }
            diff
        }

        #[allow(clippy::too_many_arguments)]
        fn run_legacy_sparse(
            ctx: &MetalContext,
            gate: &ProtocolObject<dyn MTLBuffer>,
            up: &ProtocolObject<dyn MTLBuffer>,
            down: &ProtocolObject<dyn MTLBuffer>,
            gate_expert_bytes: usize,
            up_expert_bytes: usize,
            down_expert_bytes: usize,
            token_ids: &[u32],
            route_weights: &[f32],
            slot_indices: &[usize],
            seq_len: usize,
            n_expert: usize,
            hidden_dim: usize,
            ffn_dim: usize,
            norm_all: &[f32],
            initial: Option<&[f32]>,
        ) -> LegacySparseStages {
            let token_ids_buf = shared_u32_buf(ctx, token_ids);
            let route_weights_buf = shared_f32_buf(ctx, route_weights);
            let accum = QwenMoePrefillAccumCarrier::new(ctx, hidden_dim, seq_len);
            accum.clear();
            accum.upload_norm_all(norm_all);
            if let Some(initial) = initial {
                write_f32(&accum.out_dev, initial);
            }

            let slots = seq_len * n_expert;
            let mut stages = LegacySparseStages {
                norm: vec![0.0; slots * hidden_dim],
                gate: vec![0.0; slots * ffn_dim],
                up: vec![0.0; slots * ffn_dim],
                activation: vec![0.0; slots * ffn_dim],
                down: vec![0.0; slots * hidden_dim],
                output: Vec::new(),
            };
            for expert in 0..n_expert {
                let group_start = expert * seq_len;
                let group_slots = &slot_indices[group_start..group_start + seq_len];
                let carrier = PrefillFfnCarrier::new(ctx, hidden_dim, ffn_dim, seq_len);
                submit(ctx, |encoder| {
                    prefill_ffn_chain_v2_scatter_accum_encode_gather(
                        ctx,
                        encoder,
                        &carrier,
                        &accum,
                        &accum.norm_all_dev,
                        gate,
                        (expert * gate_expert_bytes) as u32,
                        up,
                        (expert * up_expert_bytes) as u32,
                        down,
                        (expert * down_expert_bytes) as u32,
                        true,
                        seq_len,
                        &token_ids_buf,
                        &route_weights_buf,
                        group_start as u32,
                        (seq_len * hidden_dim) as u32,
                    );
                });
                align_group(
                    &mut stages.norm,
                    &readback(&carrier.normed_dev, seq_len * hidden_dim),
                    group_slots,
                    hidden_dim,
                );
                align_group(
                    &mut stages.gate,
                    &readback(&carrier.gate_dev, seq_len * ffn_dim),
                    group_slots,
                    ffn_dim,
                );
                align_group(
                    &mut stages.up,
                    &readback(&carrier.up_dev, seq_len * ffn_dim),
                    group_slots,
                    ffn_dim,
                );
                align_group(
                    &mut stages.activation,
                    &readback_f16(&carrier.act_f16_dev, seq_len * ffn_dim),
                    group_slots,
                    ffn_dim,
                );
                align_group(
                    &mut stages.down,
                    &readback(&carrier.down_dev, seq_len * hidden_dim),
                    group_slots,
                    hidden_dim,
                );
            }
            stages.output = accum.readback();
            stages
        }

        let _env_lock = ENV_LOCK.lock().unwrap();
        let _gather_f16 = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_GATHER_F16", "0");
        let _down_scatter = EnvGuard::set("RNB_METAL_QWEN35_MOE_PREFILL_DOWN_SCATTER", "0");
        let ctx = build_metal_context().expect("no metal device");
        if !ctx.tensorops_capable {
            return;
        }

        const SEQ_LEN: usize = 128;
        const N_EXPERT: usize = 8;
        const N_EXPERT_USED: usize = 8;
        const HIDDEN_DIM: usize = 2048;
        const FFN_DIM: usize = 512;
        const SLOTS: usize = SEQ_LEN * N_EXPERT_USED;

        let route_pattern = [
            f32::from_bits(0x3e61_47ae),
            f32::from_bits(0x3e38_51ec),
            f32::from_bits(0x3e19_999a),
            f32::from_bits(0x3e05_1eb8),
            f32::from_bits(0x3de1_47ae),
            f32::from_bits(0x3db8_51ec),
            f32::from_bits(0x3d8f_5c29),
            f32::from_bits(0x3d4c_cccd),
        ];
        let selected_experts = (0..SLOTS)
            .map(|slot| {
                let token = slot / N_EXPERT_USED;
                let rank = slot % N_EXPERT_USED;
                ((rank + token * 3) % N_EXPERT) as u32
            })
            .collect::<Vec<_>>();
        let route_weights = (0..SLOTS)
            .map(|slot| route_pattern[slot % N_EXPERT_USED])
            .collect::<Vec<_>>();
        let shared_route_weights = (0..SEQ_LEN)
            .map(|token| f32::from_bits(0x3e80_0000 + ((token % 7) as u32) * 0x0004_0000))
            .collect::<Vec<_>>();
        let norm_all = qwen_moe_llama_test_input(SEQ_LEN, HIDDEN_DIM);
        let norm_all_dev = shared_f32_buf(&ctx, &norm_all);

        let sparse_quant = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q6K,
        };
        let shared_quant = QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q6K,
        };
        let (gate_bytes, _) = qwen_moe_llama_test_sparse_matrix(
            sparse_quant.gate,
            N_EXPERT,
            FFN_DIM,
            HIDDEN_DIM,
            101,
        );
        let (up_bytes, _) =
            qwen_moe_llama_test_sparse_matrix(sparse_quant.up, N_EXPERT, FFN_DIM, HIDDEN_DIM, 2003);
        let (down_bytes, _) = qwen_moe_llama_test_sparse_matrix(
            sparse_quant.down,
            N_EXPERT,
            HIDDEN_DIM,
            FFN_DIM,
            4001,
        );
        let (shared_gate_bytes, _) =
            qwen_moe_llama_test_shared_matrix(shared_quant.gate, FFN_DIM, HIDDEN_DIM, 7001);
        let (shared_up_bytes, _) =
            qwen_moe_llama_test_shared_matrix(shared_quant.up, FFN_DIM, HIDDEN_DIM, 8009);
        let (shared_down_bytes, _) =
            qwen_moe_llama_test_shared_matrix(shared_quant.down, HIDDEN_DIM, FFN_DIM, 9001);
        let gate_dev = shared_u8_buf(&ctx, &gate_bytes);
        let up_dev = shared_u8_buf(&ctx, &up_bytes);
        let down_dev = shared_u8_buf(&ctx, &down_bytes);
        let shared_gate_dev = shared_u8_buf(&ctx, &shared_gate_bytes);
        let shared_up_dev = shared_u8_buf(&ctx, &shared_up_bytes);
        let shared_down_dev = shared_u8_buf(&ctx, &shared_down_bytes);
        let gate_expert_bytes = gate_bytes.len() / N_EXPERT;
        let up_expert_bytes = up_bytes.len() / N_EXPERT;
        let down_expert_bytes = down_bytes.len() / N_EXPERT;

        let QwenMoeLlamaIdPreflight::Run(plan) = qwen_moe_llama_id_preflight(
            true,
            true,
            SEQ_LEN,
            N_EXPERT,
            N_EXPERT_USED,
            HIDDEN_DIM,
            FFN_DIM,
            usize::MAX,
        )
        .expect("valid product-shape oracle plan") else {
            panic!("product-shape oracle must run");
        };
        let carrier = QwenMoeLlamaIdCarrier::new(
            &ctx,
            plan,
            &selected_experts,
            &route_weights,
            &shared_route_weights,
        )
        .expect("llama-ID carrier");
        let (uploaded_ids, uploaded_routes, uploaded_shared_routes) = carrier.debug_routes();
        assert_eq!(uploaded_ids, selected_experts);
        assert_eq!(
            uploaded_routes
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            route_weights
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            uploaded_shared_routes
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            shared_route_weights
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        submit(&ctx, |encoder| {
            qwen_moe_llama_id_map0_encode(
                &ctx,
                encoder,
                &carrier.selected_experts_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                SEQ_LEN,
                N_EXPERT_USED,
                N_EXPERT,
            )
            .expect("llama-ID map");
        });
        assert_eq!(
            readback_u32(&carrier.tpe_dev, N_EXPERT),
            vec![SEQ_LEN as u32; N_EXPERT]
        );
        let mapped_ids = readback_u32(&carrier.ids_dev, N_EXPERT * SEQ_LEN);
        let mut expected_mapped_ids = Vec::with_capacity(N_EXPERT * SEQ_LEN);
        for expert in 0..N_EXPERT {
            for token in 0..SEQ_LEN {
                let rank = selected_experts[token * N_EXPERT_USED..(token + 1) * N_EXPERT_USED]
                    .iter()
                    .position(|&id| id as usize == expert)
                    .expect("every expert selected");
                expected_mapped_ids.push((token * N_EXPERT_USED + rank) as u32);
            }
        }
        assert_eq!(mapped_ids, expected_mapped_ids);

        let gate_shape = QwenMoeLlamaIdMatmulShape {
            input_dim: HIDDEN_DIM,
            output_dim: FFN_DIM,
            n_tokens: SEQ_LEN,
            n_expert: N_EXPERT,
            n_expert_used: N_EXPERT_USED,
            expert_weight_bytes: gate_expert_bytes,
        };
        let up_shape = QwenMoeLlamaIdMatmulShape {
            expert_weight_bytes: up_expert_bytes,
            ..gate_shape
        };
        let down_shape = QwenMoeLlamaIdMatmulShape {
            input_dim: FFN_DIM,
            output_dim: HIDDEN_DIM,
            n_tokens: SEQ_LEN,
            n_expert: N_EXPERT,
            n_expert_used: 1,
            expert_weight_bytes: down_expert_bytes,
        };
        submit(&ctx, |encoder| {
            qwen_moe_llama_mul_mm_id_encode(
                &ctx,
                encoder,
                sparse_quant.gate,
                QwenMoeLlamaIdInput::F32,
                &gate_dev,
                0,
                &norm_all_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                &carrier.gate_dev,
                gate_shape,
            )
            .expect("llama-ID gate");
        });
        let id_gate = readback(&carrier.gate_dev, SLOTS * FFN_DIM);
        submit(&ctx, |encoder| {
            qwen_moe_llama_mul_mm_id_encode(
                &ctx,
                encoder,
                sparse_quant.up,
                QwenMoeLlamaIdInput::F32,
                &up_dev,
                0,
                &norm_all_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                &carrier.up_dev,
                up_shape,
            )
            .expect("llama-ID up");
        });
        let id_up = readback(&carrier.up_dev, SLOTS * FFN_DIM);
        submit(&ctx, |encoder| {
            qwen_moe_llama_swiglu_f32_encode(
                &ctx,
                encoder,
                &carrier.gate_dev,
                &carrier.up_dev,
                &carrier.gate_dev,
                SLOTS * FFN_DIM,
            )
            .expect("llama-ID activation");
        });
        let id_activation = readback(&carrier.gate_dev, SLOTS * FFN_DIM);
        submit(&ctx, |encoder| {
            qwen_moe_llama_mul_mm_id_encode(
                &ctx,
                encoder,
                sparse_quant.down,
                QwenMoeLlamaIdInput::F32,
                &down_dev,
                0,
                &carrier.gate_dev,
                &carrier.tpe_dev,
                &carrier.ids_dev,
                &carrier.down_rank_dev,
                down_shape,
            )
            .expect("llama-ID down");
        });
        let id_down = readback(&carrier.down_rank_dev, SLOTS * HIDDEN_DIM);
        submit(&ctx, |encoder| {
            qwen_moe_llama_weighted_rank_reduce_f32_encode(
                &ctx,
                encoder,
                &carrier.down_rank_dev,
                &carrier.route_weights_dev,
                &carrier.sparse_moe_out_dev,
                SEQ_LEN,
                N_EXPERT_USED,
                HIDDEN_DIM,
            )
            .expect("llama-ID sparse reduce");
        });
        let id_sparse = readback(&carrier.sparse_moe_out_dev, SEQ_LEN * HIDDEN_DIM);

        submit(&ctx, |encoder| {
            qwen_moe_shared_mul_mm_quant_f32_encode(
                &ctx,
                encoder,
                shared_quant.gate,
                &shared_gate_dev,
                0,
                &norm_all_dev,
                &carrier.gate_dev,
                HIDDEN_DIM,
                FFN_DIM,
                SEQ_LEN,
            )
            .expect("llama-ID shared gate");
        });
        let id_shared_gate = readback(&carrier.gate_dev, SEQ_LEN * FFN_DIM);
        submit(&ctx, |encoder| {
            qwen_moe_shared_mul_mm_quant_f32_encode(
                &ctx,
                encoder,
                shared_quant.up,
                &shared_up_dev,
                0,
                &norm_all_dev,
                &carrier.up_dev,
                HIDDEN_DIM,
                FFN_DIM,
                SEQ_LEN,
            )
            .expect("llama-ID shared up");
        });
        let id_shared_up = readback(&carrier.up_dev, SEQ_LEN * FFN_DIM);
        submit(&ctx, |encoder| {
            qwen_moe_llama_swiglu_f32_encode(
                &ctx,
                encoder,
                &carrier.gate_dev,
                &carrier.up_dev,
                &carrier.gate_dev,
                SEQ_LEN * FFN_DIM,
            )
            .expect("llama-ID shared activation");
        });
        submit(&ctx, |encoder| {
            qwen_moe_shared_mul_mm_quant_f32_encode(
                &ctx,
                encoder,
                shared_quant.down,
                &shared_down_dev,
                0,
                &carrier.gate_dev,
                &carrier.shared_out_dev,
                FFN_DIM,
                HIDDEN_DIM,
                SEQ_LEN,
            )
            .expect("llama-ID shared down");
        });
        let id_shared_down = readback(&carrier.shared_out_dev, SEQ_LEN * HIDDEN_DIM);
        submit(&ctx, |encoder| {
            qwen_moe_llama_weighted_rank_reduce_f32_encode(
                &ctx,
                encoder,
                &carrier.shared_out_dev,
                &carrier.shared_route_weights_dev,
                &carrier.shared_out_dev,
                SEQ_LEN,
                1,
                HIDDEN_DIM,
            )
            .expect("llama-ID shared scale");
        });
        submit(&ctx, |encoder| {
            encode_residual_add_inline_dim(
                &ctx,
                encoder,
                &carrier.sparse_moe_out_dev,
                &carrier.shared_out_dev,
                SEQ_LEN * HIDDEN_DIM,
            )
            .expect("llama-ID combined residual");
        });
        let id_combined = readback(&carrier.sparse_moe_out_dev, SEQ_LEN * HIDDEN_DIM);

        let mut legacy_token_ids = Vec::with_capacity(SLOTS);
        let mut legacy_route_weights = Vec::with_capacity(SLOTS);
        let mut legacy_slot_indices = Vec::with_capacity(SLOTS);
        for expert in 0..N_EXPERT {
            for token in 0..SEQ_LEN {
                let rank = selected_experts[token * N_EXPERT_USED..(token + 1) * N_EXPERT_USED]
                    .iter()
                    .position(|&id| id as usize == expert)
                    .expect("every expert selected");
                let slot = token * N_EXPERT_USED + rank;
                legacy_token_ids.push(token as u32);
                legacy_route_weights.push(route_weights[slot]);
                legacy_slot_indices.push(slot);
            }
        }
        let legacy_sparse = run_legacy_sparse(
            &ctx,
            &gate_dev,
            &up_dev,
            &down_dev,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            &legacy_token_ids,
            &legacy_route_weights,
            &legacy_slot_indices,
            SEQ_LEN,
            N_EXPERT,
            HIDDEN_DIM,
            FFN_DIM,
            &norm_all,
            None,
        );
        let mut expected_norm = vec![0.0; SLOTS * HIDDEN_DIM];
        for slot in 0..SLOTS {
            let token = slot / N_EXPERT_USED;
            expected_norm[slot * HIDDEN_DIM..(slot + 1) * HIDDEN_DIM]
                .copy_from_slice(&norm_all[token * HIDDEN_DIM..(token + 1) * HIDDEN_DIM]);
        }

        let shared_token_ids = (0..SEQ_LEN as u32).collect::<Vec<_>>();
        let shared_slot_indices = (0..SEQ_LEN).collect::<Vec<_>>();
        let legacy_shared = run_legacy_sparse(
            &ctx,
            &shared_gate_dev,
            &shared_up_dev,
            &shared_down_dev,
            shared_gate_bytes.len(),
            shared_up_bytes.len(),
            shared_down_bytes.len(),
            &shared_token_ids,
            &shared_route_weights,
            &shared_slot_indices,
            SEQ_LEN,
            1,
            HIDDEN_DIM,
            FFN_DIM,
            &norm_all,
            None,
        );
        let legacy_shared_weighted = &legacy_shared.output;
        let legacy_combined = run_legacy_sparse(
            &ctx,
            &gate_dev,
            &up_dev,
            &down_dev,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            &legacy_token_ids,
            &legacy_route_weights,
            &legacy_slot_indices,
            SEQ_LEN,
            N_EXPERT,
            HIDDEN_DIM,
            FFN_DIM,
            &norm_all,
            Some(legacy_shared_weighted),
        )
        .output;

        assert_eq!(
            legacy_slot_indices
                .iter()
                .map(|&slot| selected_experts[slot])
                .collect::<Vec<_>>(),
            (0..N_EXPERT)
                .flat_map(|expert| std::iter::repeat_n(expert as u32, SEQ_LEN))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            legacy_route_weights
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            legacy_slot_indices
                .iter()
                .map(|&slot| route_weights[slot].to_bits())
                .collect::<Vec<_>>()
        );

        let norm_diff = stage_diff(&expected_norm, &legacy_sparse.norm);
        let gate_diff = stage_diff(&id_gate, &legacy_sparse.gate);
        let up_diff = stage_diff(&id_up, &legacy_sparse.up);
        let activation_diff = stage_diff(&id_activation, &legacy_sparse.activation);
        let down_diff = stage_diff(&id_down, &legacy_sparse.down);
        let sparse_reduce_diff = stage_diff(&id_sparse, &legacy_sparse.output);
        let shared_gate_diff = stage_diff(&id_shared_gate, &legacy_shared.gate);
        let shared_up_diff = stage_diff(&id_shared_up, &legacy_shared.up);
        let shared_down_diff = stage_diff(&id_shared_down, &legacy_shared.down);
        let combined_diff = stage_diff(&id_combined, &legacy_combined);

        assert!(norm_diff.first_bits.is_none(), "norm: {norm_diff:?}");
        assert!(gate_diff.first_bits.is_none(), "gate: {gate_diff:?}");
        assert!(up_diff.first_bits.is_none(), "up: {up_diff:?}");
        assert!(
            activation_diff.first_bits.is_some()
                && activation_diff.max_abs <= 0.004
                && activation_diff.max_rel <= 0.0004,
            "activation: {activation_diff:?}"
        );
        assert!(down_diff.first_bits.is_none(), "down: {down_diff:?}");
        assert!(
            sparse_reduce_diff.max_abs <= 0.00025 && sparse_reduce_diff.max_rel <= 0.0000002,
            "sparse reduce: {sparse_reduce_diff:?}"
        );
        assert!(
            shared_gate_diff.first_bits.is_none(),
            "shared gate: {shared_gate_diff:?}"
        );
        assert!(
            shared_up_diff.first_bits.is_none(),
            "shared up: {shared_up_diff:?}"
        );
        assert!(
            shared_down_diff.first_bits.is_none(),
            "shared down: {shared_down_diff:?}"
        );
        assert!(
            combined_diff.max_abs <= 0.0005 && combined_diff.max_rel <= 0.0000003,
            "combined residual: {combined_diff:?}"
        );

        let mut accepted_cpu_order = vec![0.0f32; SEQ_LEN * HIDDEN_DIM];
        for token in 0..SEQ_LEN {
            for row in 0..HIDDEN_DIM {
                let mut value = legacy_sparse.down[token * N_EXPERT_USED * HIDDEN_DIM + row]
                    * route_weights[token * N_EXPERT_USED];
                for rank in 1..N_EXPERT_USED {
                    let slot = token * N_EXPERT_USED + rank;
                    let product = legacy_sparse.down[slot * HIDDEN_DIM + row] * route_weights[slot];
                    value = value + product;
                }
                let shared_product =
                    legacy_shared.down[token * HIDDEN_DIM + row] * shared_route_weights[token];
                accepted_cpu_order[token * HIDDEN_DIM + row] = value + shared_product;
            }
        }
        let legacy_vs_accepted = stage_diff(&legacy_combined, &accepted_cpu_order);
        assert!(
            legacy_vs_accepted.first_bits.is_some(),
            "expert-major shared-first legacy accumulation unexpectedly matched rank-major \
             sparse-then-shared CPU ordering"
        );
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_prefill_rank_reduce_separates_multiply_and_add() {
        let ctx = build_metal_context().expect("no metal device");
        let slot_values_cpu = [
            f32::from_bits(0xbf80_0002),
            f32::from_bits(0x3f80_0001),
            f32::from_bits(0x2900_0000),
        ];
        let weights_cpu = [1.0, f32::from_bits(0x3f80_0001), 0.5];
        let slot_values = shared_f32_buf(&ctx, &slot_values_cpu);
        let route_weights = shared_f32_buf(&ctx, &weights_cpu);
        let output = empty_f32_buf(&ctx, 1);

        let mut expected = slot_values_cpu[0] * weights_cpu[0];
        for rank in 1..slot_values_cpu.len() {
            let product = slot_values_cpu[rank] * weights_cpu[rank];
            expected = expected + product;
        }
        let rank0_product = slot_values_cpu[0] * weights_cpu[0];
        let fused = slot_values_cpu[2].mul_add(
            weights_cpu[2],
            slot_values_cpu[1].mul_add(weights_cpu[1], rank0_product),
        );
        let rank1_product = slot_values_cpu[1] * weights_cpu[1];
        let rank2_product = slot_values_cpu[2] * weights_cpu[2];
        let reassociated = rank0_product + (rank1_product + rank2_product);
        assert_eq!(expected.to_bits(), 0x2880_0000);
        assert_eq!(fused.to_bits(), 0x2900_0000);
        assert_eq!(reassociated.to_bits(), 0);

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("compute encoder");
        qwen_moe_llama_weighted_rank_reduce_f32_encode(
            &ctx,
            &encoder,
            &slot_values,
            &route_weights,
            &output,
            1,
            slot_values_cpu.len(),
            1,
        )
        .expect("strict rank reduction");
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        let got = readback(&output, 1)[0];
        assert_eq!(got.to_bits(), expected.to_bits());
        assert_ne!(got.to_bits(), fused.to_bits());
        assert_ne!(got.to_bits(), reassociated.to_bits());
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_moe_llama_sparse_reduce_matches_cpu_expert_order_bits() {
        let ctx = build_metal_context().expect("no metal device");
        let slot_values_cpu = [1.0e20f32, -1.0e20, 3.25];
        let route_weights_cpu = [1.0f32; 3];
        let selected_experts_cpu = [2u32, 0, 1];
        let slot_values = shared_f32_buf(&ctx, &slot_values_cpu);
        let route_weights = shared_f32_buf(&ctx, &route_weights_cpu);
        let selected_experts = shared_u32_buf(&ctx, &selected_experts_cpu);
        let truncated_experts = shared_u32_buf(&ctx, &selected_experts_cpu[..2]);
        let output = empty_f32_buf(&ctx, 1);

        let mut order = [0usize, 1, 2];
        order.sort_by_key(|&rank| selected_experts_cpu[rank]);
        let mut expected = 0.0f32;
        for rank in order {
            let product = route_weights_cpu[rank] * slot_values_cpu[rank];
            expected = expected + product;
        }
        let mut rank_order = 0.0f32;
        for rank in 0..slot_values_cpu.len() {
            let product = route_weights_cpu[rank] * slot_values_cpu[rank];
            rank_order = rank_order + product;
        }
        assert_eq!(expected.to_bits(), 0);
        assert_eq!(rank_order.to_bits(), 3.25f32.to_bits());

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("compute encoder");
        assert_eq!(
            qwen_moe_llama_expert_order_reduce_f32_encode(
                &ctx,
                &encoder,
                &slot_values,
                &route_weights,
                &truncated_experts,
                &output,
                1,
                3,
                1,
            ),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        assert_eq!(
            qwen_moe_llama_expert_order_reduce_f32_encode(
                &ctx,
                &encoder,
                &slot_values,
                &route_weights,
                &selected_experts,
                &output,
                1,
                9,
                1,
            ),
            Err(QwenMoeLlamaIdError::InvalidShape)
        );
        qwen_moe_llama_expert_order_reduce_f32_encode(
            &ctx,
            &encoder,
            &slot_values,
            &route_weights,
            &selected_experts,
            &output,
            1,
            3,
            1,
        )
        .expect("expert-order reduction");
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        let got = readback(&output, 1)[0];
        assert_eq!(got.to_bits(), expected.to_bits());
        assert_ne!(got.to_bits(), rank_order.to_bits());
    }

    fn iq2xxs_fixture(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let blocks_per_row = cols / 256;
        let mut weights = vec![0u8; rows * blocks_per_row * 66];
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let base = (row * blocks_per_row + block) * 66;
                weights[base..base + 2].copy_from_slice(
                    &half::f16::from_f32(0.0015 + ((row + block + seed) % 7) as f32 * 0.0002)
                        .to_bits()
                        .to_le_bytes(),
                );
                for subblock in 0..8 {
                    let packed = base + 2 + subblock * 8;
                    for index in 0..4 {
                        weights[packed + index] =
                            ((row * 31 + block * 17 + subblock * 11 + index * 43 + seed) & 0xff)
                                as u8;
                    }
                    let mut aux = (((row + block + subblock + seed) % 16) as u32) << 28;
                    for sign_group in 0..4 {
                        let signs =
                            ((row * 13 + block * 19 + subblock * 7 + sign_group * 23 + seed) & 127)
                                as u32;
                        aux |= signs << (7 * sign_group);
                    }
                    weights[packed + 4..packed + 8].copy_from_slice(&aux.to_le_bytes());
                }
            }
        }
        weights
    }

    fn iq3xxs_fixture(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let blocks_per_row = cols / 256;
        let mut weights = vec![0u8; rows * blocks_per_row * 98];
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let base = (row * blocks_per_row + block) * 98;
                weights[base..base + 2].copy_from_slice(
                    &half::f16::from_f32(0.0012 + ((row + block + seed) % 5) as f32 * 0.00025)
                        .to_bits()
                        .to_le_bytes(),
                );
                for index in 0..64 {
                    weights[base + 2 + index] =
                        ((row * 29 + block * 37 + index * 41 + seed) & 0xff) as u8;
                }
                for subblock in 0..8 {
                    let mut aux = (((row + block + subblock + seed) % 16) as u32) << 28;
                    for sign_group in 0..4 {
                        let signs =
                            ((row * 17 + block * 11 + subblock * 5 + sign_group * 31 + seed) & 127)
                                as u32;
                        aux |= signs << (7 * sign_group);
                    }
                    let gas = base + 66 + subblock * 4;
                    weights[gas..gas + 4].copy_from_slice(&aux.to_le_bytes());
                }
            }
        }
        weights
    }

    fn iq4xs_fixture(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let blocks_per_row = cols / 256;
        let mut weights = vec![0u8; rows * blocks_per_row * 136];
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let base = (row * blocks_per_row + block) * 136;
                weights[base..base + 2].copy_from_slice(
                    &half::f16::from_f32(0.0018 + ((row + block + seed) % 6) as f32 * 0.0003)
                        .to_bits()
                        .to_le_bytes(),
                );
                let scales_h = ((row * 23 + block * 29 + seed) & 0xffff) as u16;
                weights[base + 2..base + 4].copy_from_slice(&scales_h.to_le_bytes());
                for index in 0..4 {
                    weights[base + 4 + index] =
                        ((row * 13 + block * 31 + index * 47 + seed) & 0xff) as u8;
                }
                for index in 0..128 {
                    weights[base + 8 + index] =
                        ((row * 37 + block * 19 + index * 53 + seed) & 0xff) as u8;
                }
            }
        }
        weights
    }

    fn iq2s_fixture(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let blocks_per_row = cols / 256;
        let mut weights = vec![0u8; rows * blocks_per_row * 82];
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let base = (row * blocks_per_row + block) * 82;
                weights[base..base + 2].copy_from_slice(
                    &half::f16::from_f32(0.0016 + ((row + block + seed) % 9) as f32 * 0.00021)
                        .to_bits()
                        .to_le_bytes(),
                );
                for index in 0..64 {
                    weights[base + 2 + index] =
                        ((row * 41 + block * 23 + index * 59 + seed) & 0xff) as u8;
                }
                for index in 0..8 {
                    weights[base + 66 + index] =
                        ((row * 11 + block * 43 + index * 61 + seed) & 0xff) as u8;
                }
                for index in 0..8 {
                    weights[base + 74 + index] =
                        ((row * 53 + block * 7 + index * 29 + seed) & 0xff) as u8;
                }
            }
        }
        weights
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn glm_moe_iq_selected_slots_match_cpu_oracle() {
        use rnb_cpu::gemm::quant_gemv::{dot_quantized_row, QuantGemvType};

        let ctx = build_metal_context().expect("no metal device");
        let rows = 17usize;
        let cols = 512usize;
        let slots = 2usize;
        let input: Vec<f32> = (0..cols)
            .map(|index| ((index * 17 + 5) % 101) as f32 * 0.005 - 0.25)
            .collect();
        let slot_input: Vec<f32> = (0..slots)
            .flat_map(|slot| input.iter().map(move |value| value + slot as f32 * 0.03125))
            .collect();
        let iq2_weights = (0..slots)
            .map(|slot| iq2xxs_fixture(rows, cols, slot * 7))
            .collect::<Vec<_>>();
        let iq3_weights = (0..slots)
            .map(|slot| iq3xxs_fixture(rows, cols, slot * 11))
            .collect::<Vec<_>>();
        let iq4_weights = (0..slots)
            .map(|slot| iq4xs_fixture(rows, cols, slot * 13))
            .collect::<Vec<_>>();
        let iq2s_weights = (0..slots)
            .map(|slot| iq2s_fixture(rows, cols, slot * 17))
            .collect::<Vec<_>>();
        let iq2_buffers = iq2_weights
            .iter()
            .map(|weight| shared_u8_buf(&ctx, weight))
            .collect::<Vec<_>>();
        let iq3_buffers = iq3_weights
            .iter()
            .map(|weight| shared_u8_buf(&ctx, weight))
            .collect::<Vec<_>>();
        let iq4_buffers = iq4_weights
            .iter()
            .map(|weight| shared_u8_buf(&ctx, weight))
            .collect::<Vec<_>>();
        let iq2s_buffers = iq2s_weights
            .iter()
            .map(|weight| shared_u8_buf(&ctx, weight))
            .collect::<Vec<_>>();
        let offsets = vec![0u32; slots];
        let input_buf = shared_f32_buf(&ctx, &input);
        let slot_input_buf = shared_f32_buf(&ctx, &slot_input);
        let iq2_out = empty_f32_buf(&ctx, rows * slots);
        let iq3_out = empty_f32_buf(&ctx, rows * slots);
        let iq4_out = empty_f32_buf(&ctx, rows * slots);
        let iq2s_out = empty_f32_buf(&ctx, rows * slots);
        let rows_buf = u32_buf(&ctx, rows as u32);
        let cols_buf = u32_buf(&ctx, cols as u32);
        let slots_buf = u32_buf(&ctx, slots as u32);

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("compute encoder");
        encode_glm_moe_decode_iq_selected_slots(
            ctx.glm_moe_decode_iq2xxs_selected_slots_pipeline(),
            &encoder,
            &iq2_buffers,
            &offsets,
            &input_buf,
            &iq2_out,
            &rows_buf,
            &cols_buf,
            &slots_buf,
            rows,
            slots,
        );
        encode_glm_moe_decode_iq_selected_slots(
            ctx.glm_moe_decode_iq3xxs_selected_slots_pipeline(),
            &encoder,
            &iq3_buffers,
            &offsets,
            &slot_input_buf,
            &iq3_out,
            &rows_buf,
            &cols_buf,
            &slots_buf,
            rows,
            slots,
        );
        encode_glm_moe_decode_iq_selected_slots(
            ctx.glm_moe_decode_iq4xs_selected_slots_pipeline(),
            &encoder,
            &iq4_buffers,
            &offsets,
            &slot_input_buf,
            &iq4_out,
            &rows_buf,
            &cols_buf,
            &slots_buf,
            rows,
            slots,
        );
        encode_glm_moe_decode_iq_selected_slots(
            ctx.glm_moe_decode_iq2s_selected_slots_pipeline(),
            &encoder,
            &iq2s_buffers,
            &offsets,
            &input_buf,
            &iq2s_out,
            &rows_buf,
            &cols_buf,
            &slots_buf,
            rows,
            slots,
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        let iq2_actual = readback(&iq2_out, rows * slots);
        let iq3_actual = readback(&iq3_out, rows * slots);
        let iq4_actual = readback(&iq4_out, rows * slots);
        let iq2s_actual = readback(&iq2s_out, rows * slots);
        for slot in 0..slots {
            for row in 0..rows {
                let iq2_row_bytes = cols / 256 * 66;
                let iq3_row_bytes = cols / 256 * 98;
                let iq2_expected = dot_quantized_row(
                    &iq2_weights[slot][row * iq2_row_bytes..(row + 1) * iq2_row_bytes],
                    &input,
                    cols,
                    QuantGemvType::IQ2XXS,
                );
                let iq3_expected = dot_quantized_row(
                    &iq3_weights[slot][row * iq3_row_bytes..(row + 1) * iq3_row_bytes],
                    &slot_input[slot * cols..(slot + 1) * cols],
                    cols,
                    QuantGemvType::IQ3XXS,
                );
                let iq4_row_bytes = cols / 256 * 136;
                let iq4_dequant = rnb_cpu::gemm::dequant::dequantize_bytes_to_f32(
                    &iq4_weights[slot][row * iq4_row_bytes..(row + 1) * iq4_row_bytes],
                    rnb_cpu::gemm::dequant::DequantType::IQ4XS,
                );
                let iq4_expected: f32 = iq4_dequant
                    .iter()
                    .zip(&slot_input[slot * cols..(slot + 1) * cols])
                    .map(|(w, x)| w * x)
                    .sum();
                let iq2s_row_bytes = cols / 256 * 82;
                let iq2s_dequant = rnb_cpu::gemm::dequant::dequantize_bytes_to_f32(
                    &iq2s_weights[slot][row * iq2s_row_bytes..(row + 1) * iq2s_row_bytes],
                    rnb_cpu::gemm::dequant::DequantType::IQ2S,
                );
                let iq2s_expected: f32 = iq2s_dequant.iter().zip(&input).map(|(w, x)| w * x).sum();
                for (label, actual, expected) in [
                    ("IQ2_XXS", iq2_actual[slot * rows + row], iq2_expected),
                    ("IQ3_XXS", iq3_actual[slot * rows + row], iq3_expected),
                    ("IQ4_XS", iq4_actual[slot * rows + row], iq4_expected),
                    ("IQ2_S", iq2s_actual[slot * rows + row], iq2s_expected),
                ] {
                    let abs = (actual - expected).abs();
                    let rel = abs / expected.abs().max(1e-5);
                    assert!(
                        abs <= 0.0005 || rel <= 0.0001,
                        "{label} slot={slot} row={row} actual={actual} expected={expected} abs={abs} rel={rel}"
                    );
                }
            }
        }
    }
}
