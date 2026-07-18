//! pm44 M1-2: GDN prefill gated→ssm_out fused device-resident. gated_rmsnorm_silu → cast → Q5_K GEMM_v2
//! 를 단일 command buffer 에 묶어 host 왕복(gated readback + ssm_out upload + commit/wait 1회) 제거.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLResourceOptions,
};

use crate::compute::{self, MetalContext};
pub(crate) use crate::ffn_chain::{empty_f16_buf, empty_f32_buf, u32_buf};
pub(crate) use crate::gdn_proj_chain::TensoropsQuant;

pub(crate) struct PrefillGdnGatedProjCarrier {
    pub seq_len: usize,
    pub d_inner: usize,    // = num_v_heads * head_v_dim
    pub head_v_dim: usize, // gated cols
    pub n_out: usize,      // ssm_out proj 출력 = weight.rows()
    pub out_in_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*d_inner] scan 출력
    pub z_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*d_inner] gate
    pub norm_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [head_v_dim] ssm_norm
    pub gated_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*d_inner] gated 출력 = GEMM 입력
    pub gated_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*d_inner] half
    pub gated_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(seq*d_inner)
    pub out_dev: Retained<ProtocolObject<dyn MTLBuffer>>, // [seq*n_out]
    pub cols_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(head_v_dim)
    pub eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // f32(norm_eps)
    pub n_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(n_out)
    pub k_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(d_inner)
    pub m_buf: Retained<ProtocolObject<dyn MTLBuffer>>, // u32(seq)
}

impl PrefillGdnGatedProjCarrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        d_inner: usize,
        head_v_dim: usize,
        n_out: usize,
        norm_eps: f32,
    ) -> Self {
        debug_assert_eq!(
            d_inner % head_v_dim,
            0,
            "d_inner must be num_v_heads*head_v_dim"
        );
        let eps_buf = unsafe {
            let p =
                std::ptr::NonNull::new(&norm_eps as *const f32 as *mut std::ffi::c_void).unwrap();
            ctx.device
                .newBufferWithBytes_length_options(p, 4, MTLResourceOptions::StorageModeShared)
                .expect("eps buf")
        };
        Self {
            seq_len,
            d_inner,
            head_v_dim,
            n_out,
            out_in_dev: empty_f32_buf(ctx, seq_len * d_inner),
            z_dev: empty_f32_buf(ctx, seq_len * d_inner),
            norm_dev: empty_f32_buf(ctx, head_v_dim),
            gated_dev: empty_f32_buf(ctx, seq_len * d_inner),
            gated_f16_dev: empty_f16_buf(ctx, seq_len * d_inner),
            gated_elems_buf: u32_buf(ctx, (seq_len * d_inner) as u32),
            out_dev: empty_f32_buf(ctx, seq_len * n_out),
            cols_buf: u32_buf(ctx, head_v_dim as u32),
            eps_buf,
            n_buf: u32_buf(ctx, n_out as u32),
            k_buf: u32_buf(ctx, d_inner as u32),
            m_buf: u32_buf(ctx, seq_len as u32),
        }
    }

    fn upload(&self, out_in: &[f32], z: &[f32], norm: &[f32]) {
        debug_assert_eq!(out_in.len(), self.seq_len * self.d_inner);
        debug_assert_eq!(z.len(), self.seq_len * self.d_inner);
        debug_assert_eq!(norm.len(), self.head_v_dim);
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_in.as_ptr(),
                self.out_in_dev.contents().as_ptr() as *mut f32,
                out_in.len(),
            );
            std::ptr::copy_nonoverlapping(
                z.as_ptr(),
                self.z_dev.contents().as_ptr() as *mut f32,
                z.len(),
            );
            std::ptr::copy_nonoverlapping(
                norm.as_ptr(),
                self.norm_dev.contents().as_ptr() as *mut f32,
                norm.len(),
            );
        }
    }
}

/// pm44 M1-2: gated_rmsnorm_silu → cast f32→f16 → ssm_out GEMM_v2 를 단일 MTLCommandEncoder 에
/// 연속 encode. RAW hazard 자동 직렬화, host 왕복 1회(commit/wait/readback).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_gated_norm_silu_project_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillGdnGatedProjCarrier,
    out_in: &[f32],
    z: &[f32],
    norm: &[f32],
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    off_buf: &ProtocolObject<dyn MTLBuffer>,
    quant: TensoropsQuant,
) -> Vec<f32> {
    carrier.upload(out_in, z, norm);
    let off = unsafe { *(off_buf.contents().as_ptr() as *const u32) };
    let m = carrier.seq_len;
    let rows = carrier.seq_len * (carrier.d_inner / carrier.head_v_dim); // seq * num_v_heads
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder"); // 단일 encoder = RAW hazard 자동 직렬화
    compute::encode_gated_rmsnorm_silu(
        ctx,
        &enc,
        &carrier.out_in_dev,
        &carrier.z_dev,
        &carrier.norm_dev,
        &carrier.gated_dev,
        &carrier.cols_buf,
        &carrier.eps_buf,
        rows,
    );
    compute::encode_cast_f32_to_f16(
        ctx,
        &enc,
        &carrier.gated_dev,
        &carrier.gated_f16_dev,
        &carrier.gated_elems_buf,
        m * carrier.d_inner,
    );
    match quant {
        TensoropsQuant::Q5K => compute::encode_gemm_q5k_tensorops_v2(
            ctx,
            &enc,
            w_buf,
            off,
            &carrier.gated_f16_dev,
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
            &carrier.gated_f16_dev,
            &carrier.out_dev,
            &carrier.n_buf,
            &carrier.k_buf,
            &carrier.m_buf,
            carrier.n_out,
            m,
        ),
        TensoropsQuant::Q6K => compute::encode_gemm_q6k_tensorops_v2(
            ctx,
            &enc,
            w_buf,
            off,
            &carrier.gated_f16_dev,
            &carrier.out_dev,
            &carrier.n_buf,
            &carrier.k_buf,
            &carrier.m_buf,
            carrier.n_out,
            m,
        ),
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let c = carrier.out_dev.contents();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, m * carrier.n_out) }.to_vec()
}
