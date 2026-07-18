//! MoE section layout helpers shared by loaders and runtime adapters.

use crate::quantize::moe_blocks::{
    GUPairQ4K, GUPairQ4KScaleMin, GUPairQ4KUnpackedScales, Q5KIntScale, Q80IntScale,
    SharedGUQ8KUnit,
};

pub const GU_PAIR_Q4K_BYTES: usize = std::mem::size_of::<GUPairQ4K>();
pub const GU_PAIR_Q4K_UNPACKED_SCALES_BYTES: usize = std::mem::size_of::<GUPairQ4KUnpackedScales>();
pub const GU_PAIR_Q4K_SCALE_MIN_BYTES: usize = std::mem::size_of::<GUPairQ4KScaleMin>();
pub const DOWN_Q5K_INT_SCALE_BYTES: usize = std::mem::size_of::<Q5KIntScale>();
pub const SHARED_GU_Q8K_UNIT_BYTES: usize = std::mem::size_of::<SharedGUQ8KUnit>();
pub const SHARED_DOWN_Q80_INT_SCALE_BYTES: usize = std::mem::size_of::<Q80IntScale>();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateUpRowLayout {
    Q4KPair,
    UnpackedScales,
    ScalePlane,
}

impl GateUpRowLayout {
    #[inline]
    pub const fn unit_size(self) -> usize {
        match self {
            Self::Q4KPair | Self::ScalePlane => GU_PAIR_Q4K_BYTES,
            Self::UnpackedScales => GU_PAIR_Q4K_UNPACKED_SCALES_BYTES,
        }
    }

    #[inline]
    pub const fn uses_scale_plane(self) -> bool {
        matches!(self, Self::ScalePlane)
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_gate_up_row_q4k(
    layout: GateUpRowLayout,
    row_bytes: &[u8],
    scale_bytes: Option<&[u8]>,
    h_q8k: &[crate::gemm::Q8KBlock],
) -> (f32, f32) {
    let n_blocks = h_q8k.len();
    debug_assert_eq!(row_bytes.len(), n_blocks * layout.unit_size());

    match layout {
        GateUpRowLayout::UnpackedScales => {
            let row = unsafe {
                std::slice::from_raw_parts(
                    row_bytes.as_ptr() as *const GUPairQ4KUnpackedScales,
                    n_blocks,
                )
            };
            dot_gate_up_unpacked_scales(row, h_q8k)
        }
        GateUpRowLayout::ScalePlane => {
            let scale_bytes = scale_bytes.expect("scale-plane row missing scale bytes");
            debug_assert_eq!(scale_bytes.len(), n_blocks * GU_PAIR_Q4K_SCALE_MIN_BYTES);
            let row = unsafe {
                std::slice::from_raw_parts(row_bytes.as_ptr() as *const GUPairQ4K, n_blocks)
            };
            let scale = unsafe {
                std::slice::from_raw_parts(
                    scale_bytes.as_ptr() as *const GUPairQ4KScaleMin,
                    n_blocks,
                )
            };
            dot_gate_up_scale_plane(row, scale, h_q8k)
        }
        GateUpRowLayout::Q4KPair => {
            debug_assert!(scale_bytes.is_none());
            let row = unsafe {
                std::slice::from_raw_parts(row_bytes.as_ptr() as *const GUPairQ4K, n_blocks)
            };
            dot_gate_up_pair(row, h_q8k)
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_down_row_q5k(row_bytes: &[u8], h_q8k: &[crate::gemm::Q8KBlock]) -> f32 {
    use crate::gemm::neon_moe::sdot_q5k_row_block_neon;

    let n_blocks = h_q8k.len();
    debug_assert_eq!(row_bytes.len(), n_blocks * DOWN_Q5K_INT_SCALE_BYTES);
    let row =
        unsafe { std::slice::from_raw_parts(row_bytes.as_ptr() as *const Q5KIntScale, n_blocks) };

    if n_blocks == 2 {
        let a0 = unsafe { sdot_q5k_row_block_neon(&row[0], &h_q8k[0]) };
        let a1 = unsafe { sdot_q5k_row_block_neon(&row[1], &h_q8k[1]) };
        return (a0 as f32) * h_q8k[0].d + (a1 as f32) * h_q8k[1].d;
    }

    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let acc_int = unsafe { sdot_q5k_row_block_neon(&row[b], &h_q8k[b]) };
        acc += (acc_int as f32) * h_q8k[b].d;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_shared_gate_up_row_q80(
    row_bytes: &[u8],
    h_q8k: &[crate::gemm::Q8KBlock],
) -> (f32, f32) {
    use crate::gemm::neon_moe::sdot_q80_gu_block_neon;

    let n_units = h_q8k.len();
    debug_assert_eq!(row_bytes.len(), n_units * SHARED_GU_Q8K_UNIT_BYTES);
    let row = unsafe {
        std::slice::from_raw_parts(row_bytes.as_ptr() as *const SharedGUQ8KUnit, n_units)
    };

    if n_units == 8 {
        let (g0, u0) = unsafe { sdot_q80_gu_block_neon(&row[0], &h_q8k[0]) };
        let (g1, u1) = unsafe { sdot_q80_gu_block_neon(&row[1], &h_q8k[1]) };
        let (g2, u2) = unsafe { sdot_q80_gu_block_neon(&row[2], &h_q8k[2]) };
        let (g3, u3) = unsafe { sdot_q80_gu_block_neon(&row[3], &h_q8k[3]) };
        let (g4, u4) = unsafe { sdot_q80_gu_block_neon(&row[4], &h_q8k[4]) };
        let (g5, u5) = unsafe { sdot_q80_gu_block_neon(&row[5], &h_q8k[5]) };
        let (g6, u6) = unsafe { sdot_q80_gu_block_neon(&row[6], &h_q8k[6]) };
        let (g7, u7) = unsafe { sdot_q80_gu_block_neon(&row[7], &h_q8k[7]) };
        return fold_gate_up_8(
            [g0, g1, g2, g3, g4, g5, g6, g7],
            [u0, u1, u2, u3, u4, u5, u6, u7],
            h_q8k,
        );
    }

    let mut g_acc = 0.0f32;
    let mut u_acc = 0.0f32;
    for b in 0..n_units {
        let (g_int, u_int) = unsafe { sdot_q80_gu_block_neon(&row[b], &h_q8k[b]) };
        g_acc += (g_int as f32) * h_q8k[b].d;
        u_acc += (u_int as f32) * h_q8k[b].d;
    }
    (g_acc, u_acc)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_shared_down_row_q80(row_bytes: &[u8], h_q8k: &[crate::gemm::Q8KBlock]) -> f32 {
    use crate::gemm::neon_moe::sdot_q80_row_block_neon;

    let n_blocks = h_q8k.len();
    debug_assert_eq!(
        row_bytes.len(),
        n_blocks * 8 * SHARED_DOWN_Q80_INT_SCALE_BYTES
    );
    let row = unsafe {
        std::slice::from_raw_parts(row_bytes.as_ptr() as *const Q80IntScale, n_blocks * 8)
    };

    if n_blocks == 2 {
        let a0 = unsafe { sdot_q80_row_block_neon(&row[..8], &h_q8k[0]) };
        let a1 = unsafe { sdot_q80_row_block_neon(&row[8..16], &h_q8k[1]) };
        return (a0 as f32) * h_q8k[0].d + (a1 as f32) * h_q8k[1].d;
    }

    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let acc_int = unsafe { sdot_q80_row_block_neon(&row[b * 8..(b + 1) * 8], &h_q8k[b]) };
        acc += (acc_int as f32) * h_q8k[b].d;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_gate_up_unpacked_scales(
    row: &[GUPairQ4KUnpackedScales],
    h_q8k: &[crate::gemm::Q8KBlock],
) -> (f32, f32) {
    use crate::gemm::neon_moe::sdot_q4k_gu_block_unpacked_scales_neon;

    if row.len() == 8 {
        let (g0, u0) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[0], &h_q8k[0]) };
        let (g1, u1) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[1], &h_q8k[1]) };
        let (g2, u2) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[2], &h_q8k[2]) };
        let (g3, u3) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[3], &h_q8k[3]) };
        let (g4, u4) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[4], &h_q8k[4]) };
        let (g5, u5) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[5], &h_q8k[5]) };
        let (g6, u6) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[6], &h_q8k[6]) };
        let (g7, u7) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[7], &h_q8k[7]) };
        return fold_gate_up_8(
            [g0, g1, g2, g3, g4, g5, g6, g7],
            [u0, u1, u2, u3, u4, u5, u6, u7],
            h_q8k,
        );
    }

    let mut g_acc = 0.0f32;
    let mut u_acc = 0.0f32;
    for b in 0..row.len() {
        let (g_int, u_int) = unsafe { sdot_q4k_gu_block_unpacked_scales_neon(&row[b], &h_q8k[b]) };
        g_acc += (g_int as f32) * h_q8k[b].d;
        u_acc += (u_int as f32) * h_q8k[b].d;
    }
    (g_acc, u_acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_gate_up_scale_plane(
    row: &[GUPairQ4K],
    scale: &[GUPairQ4KScaleMin],
    h_q8k: &[crate::gemm::Q8KBlock],
) -> (f32, f32) {
    use crate::gemm::neon_moe::sdot_q4k_gu_block_scale_min_neon;

    if row.len() == 8 {
        let (g0, u0) = unsafe { sdot_q4k_gu_block_scale_min_neon(&row[0], &scale[0], &h_q8k[0]) };
        let (g1, u1) = unsafe { sdot_q4k_gu_block_scale_min_neon(&row[1], &scale[1], &h_q8k[1]) };
        let (g2, u2) = unsafe { sdot_q4k_gu_block_scale_min_neon(&row[2], &scale[2], &h_q8k[2]) };
        let (g3, u3) = unsafe { sdot_q4k_gu_block_scale_min_neon(&row[3], &scale[3], &h_q8k[3]) };
        let (g4, u4) = unsafe { sdot_q4k_gu_block_scale_min_neon(&row[4], &scale[4], &h_q8k[4]) };
        let (g5, u5) = unsafe { sdot_q4k_gu_block_scale_min_neon(&row[5], &scale[5], &h_q8k[5]) };
        let (g6, u6) = unsafe { sdot_q4k_gu_block_scale_min_neon(&row[6], &scale[6], &h_q8k[6]) };
        let (g7, u7) = unsafe { sdot_q4k_gu_block_scale_min_neon(&row[7], &scale[7], &h_q8k[7]) };
        return fold_gate_up_8(
            [g0, g1, g2, g3, g4, g5, g6, g7],
            [u0, u1, u2, u3, u4, u5, u6, u7],
            h_q8k,
        );
    }

    let mut g_acc = 0.0f32;
    let mut u_acc = 0.0f32;
    for b in 0..row.len() {
        let (g_int, u_int) =
            unsafe { sdot_q4k_gu_block_scale_min_neon(&row[b], &scale[b], &h_q8k[b]) };
        g_acc += (g_int as f32) * h_q8k[b].d;
        u_acc += (u_int as f32) * h_q8k[b].d;
    }
    (g_acc, u_acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_gate_up_pair(row: &[GUPairQ4K], h_q8k: &[crate::gemm::Q8KBlock]) -> (f32, f32) {
    use crate::gemm::neon_moe::sdot_q4k_gu_block_neon;

    if row.len() == 8 {
        let (g0, u0) = unsafe { sdot_q4k_gu_block_neon(&row[0], &h_q8k[0]) };
        let (g1, u1) = unsafe { sdot_q4k_gu_block_neon(&row[1], &h_q8k[1]) };
        let (g2, u2) = unsafe { sdot_q4k_gu_block_neon(&row[2], &h_q8k[2]) };
        let (g3, u3) = unsafe { sdot_q4k_gu_block_neon(&row[3], &h_q8k[3]) };
        let (g4, u4) = unsafe { sdot_q4k_gu_block_neon(&row[4], &h_q8k[4]) };
        let (g5, u5) = unsafe { sdot_q4k_gu_block_neon(&row[5], &h_q8k[5]) };
        let (g6, u6) = unsafe { sdot_q4k_gu_block_neon(&row[6], &h_q8k[6]) };
        let (g7, u7) = unsafe { sdot_q4k_gu_block_neon(&row[7], &h_q8k[7]) };
        return fold_gate_up_8(
            [g0, g1, g2, g3, g4, g5, g6, g7],
            [u0, u1, u2, u3, u4, u5, u6, u7],
            h_q8k,
        );
    }

    let mut g_acc = 0.0f32;
    let mut u_acc = 0.0f32;
    for b in 0..row.len() {
        let (g_int, u_int) = unsafe { sdot_q4k_gu_block_neon(&row[b], &h_q8k[b]) };
        g_acc += (g_int as f32) * h_q8k[b].d;
        u_acc += (u_int as f32) * h_q8k[b].d;
    }
    (g_acc, u_acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fold_gate_up_8(g: [i64; 8], u: [i64; 8], h_q8k: &[crate::gemm::Q8KBlock]) -> (f32, f32) {
    let gate = (g[0] as f32) * h_q8k[0].d
        + (g[1] as f32) * h_q8k[1].d
        + (g[2] as f32) * h_q8k[2].d
        + (g[3] as f32) * h_q8k[3].d
        + (g[4] as f32) * h_q8k[4].d
        + (g[5] as f32) * h_q8k[5].d
        + (g[6] as f32) * h_q8k[6].d
        + (g[7] as f32) * h_q8k[7].d;
    let up = (u[0] as f32) * h_q8k[0].d
        + (u[1] as f32) * h_q8k[1].d
        + (u[2] as f32) * h_q8k[2].d
        + (u[3] as f32) * h_q8k[3].d
        + (u[4] as f32) * h_q8k[4].d
        + (u[5] as f32) * h_q8k[5].d
        + (u[6] as f32) * h_q8k[6].d
        + (u[7] as f32) * h_q8k[7].d;
    (gate, up)
}
