use half::f16;

// Basic quant blocks (block_size = 32 elements)
pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

// K-quant super-block size
pub const QK_K: usize = 256;

#[repr(C)]
pub struct BlockQ4_0 {
    pub d: f16,              // delta (scale)
    pub qs: [u8; QK4_0 / 2], // 16 bytes: 32 × 4-bit packed
}
// size = 2 + 16 = 18

#[repr(C)]
pub struct BlockQ4_1 {
    pub d: f16, // delta
    pub m: f16, // min
    pub qs: [u8; QK4_1 / 2],
}
// size = 2 + 2 + 16 = 20

#[repr(C)]
pub struct BlockQ5_0 {
    pub d: f16,
    pub qh: [u8; 4], // 32 high bits
    pub qs: [u8; QK5_0 / 2],
}
// size = 2 + 4 + 16 = 22

#[repr(C)]
pub struct BlockQ5_1 {
    pub d: f16,
    pub m: f16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_1 / 2],
}
// size = 2 + 2 + 4 + 16 = 24

#[repr(C)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK8_0],
}
// size = 2 + 32 = 34

#[repr(C)]
pub struct BlockQ8_1 {
    pub d: f16,
    pub s: f16, // sum of quants (for dot product optimization)
    pub qs: [i8; QK8_1],
}
// size = 2 + 2 + 32 = 36

// K-Quant blocks (super-block = 256 elements)

#[repr(C)]
pub struct BlockQ2_K {
    pub scales: [u8; QK_K / 16], // 16 bytes: scales and mins (4-bit each)
    pub qs: [u8; QK_K / 4],      // 64 bytes: 2-bit quants
    pub d: f16,
    pub dmin: f16,
}
// size = 16 + 64 + 2 + 2 = 84

#[repr(C)]
pub struct BlockQ3_K {
    pub hmask: [u8; QK_K / 8], // 32 bytes: high bits
    pub qs: [u8; QK_K / 4],    // 64 bytes: low 2 bits
    pub scales: [u8; 12],      // scales (6-bit packed)
    pub d: f16,
}
// size = 32 + 64 + 12 + 2 = 110

#[repr(C)]
pub struct BlockQ4_K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],   // sub-block scales (6-bit packed)
    pub qs: [u8; QK_K / 2], // 128 bytes: 4-bit quants
}
// size = 2 + 2 + 12 + 128 = 144

#[repr(C)]
pub struct BlockQ5_K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    pub qh: [u8; QK_K / 8], // 32 bytes: high bits
    pub qs: [u8; QK_K / 2], // 128 bytes: low 4 bits
}
// size = 2 + 2 + 12 + 32 + 128 = 176

#[repr(C)]
pub struct BlockQ6_K {
    pub ql: [u8; QK_K / 2],      // 128 bytes: low 4 bits
    pub qh: [u8; QK_K / 4],      // 64 bytes: high 2 bits
    pub scales: [i8; QK_K / 16], // 16 bytes: scales
    pub d: f16,
}
// size = 128 + 64 + 16 + 2 = 210

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_basic_block_sizes() {
        assert_eq!(size_of::<BlockQ4_0>(), 18);
        assert_eq!(size_of::<BlockQ4_1>(), 20);
        assert_eq!(size_of::<BlockQ5_0>(), 22);
        assert_eq!(size_of::<BlockQ5_1>(), 24);
        assert_eq!(size_of::<BlockQ8_0>(), 34);
        assert_eq!(size_of::<BlockQ8_1>(), 36);
    }

    #[test]
    fn test_k_quant_block_sizes() {
        assert_eq!(size_of::<BlockQ2_K>(), 84);
        assert_eq!(size_of::<BlockQ3_K>(), 110);
        assert_eq!(size_of::<BlockQ4_K>(), 144);
        assert_eq!(size_of::<BlockQ5_K>(), 176);
        assert_eq!(size_of::<BlockQ6_K>(), 210);
    }

    #[test]
    fn test_block_constants() {
        assert_eq!(QK4_0, 32);
        assert_eq!(QK8_0, 32);
        assert_eq!(QK_K, 256);
    }
}
