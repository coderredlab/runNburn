//! MoE decode IntScale block structs: integer-scale prequant of Q4_K/Q5_K/Q8_0.
//! Row-level `row_multiplier: f32` is stored outside (per-row metadata).

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Q4KIntScale {
    pub scale_int: i8,        // super-block d, scaled by row_multiplier
    pub min_int: i8,          // super-block dmin, scaled by row_multiplier
    pub sub_scales: [u8; 12], // GGUF 6-bit packed per-sub-block scales (verbatim from BlockQ4_K.scales)
    pub qs: [u8; 128],        // Q4_K 256-elem block, 4 bits/elem (verbatim)
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Q5KIntScale {
    pub scale_int: i8,
    pub min_int: i8,
    pub sub_scales: [u8; 12], // GGUF 6-bit packed sub-block scales (verbatim from BlockQ5_K.scales)
    pub qs_low: [u8; 128],    // 4 LSB per elem (verbatim from BlockQ5_K.qs)
    pub qs_high: [u8; 32],    // 1 MSB per elem, 8 elems/byte (verbatim from BlockQ5_K.qh)
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Q80IntScale {
    pub scale_int: i16,
    pub qs: [i8; 32], // Q8_0 32-elem block
}

/// MoE gate/up block-interleaved unit (Q4_K).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GUPairQ4K {
    pub gate: Q4KIntScale,
    pub up: Q4KIntScale,
}

/// MoE gate/up Q4_K unit with pre-unpacked 6-bit sub-block scale/min arrays.
///
/// This is a layout experiment for MoE decode sidecars: it preserves the
/// original `GUPairQ4K` bytes and appends four 8-byte arrays so the hot decode
/// kernel does not have to unpack GGUF Q4_K `sub_scales` on every call.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GUPairQ4KUnpackedScales {
    pub pair: GUPairQ4K,
    pub gate_sc: [u8; 8],
    pub gate_m: [u8; 8],
    pub up_sc: [u8; 8],
    pub up_m: [u8; 8],
}

impl GUPairQ4KUnpackedScales {
    #[inline]
    pub fn from_pair(pair: GUPairQ4K) -> Self {
        let (gate_sc, gate_m) = unpack_k4_scales(&pair.gate.sub_scales);
        let (up_sc, up_m) = unpack_k4_scales(&pair.up.sub_scales);
        Self {
            pair,
            gate_sc,
            gate_m,
            up_sc,
            up_m,
        }
    }

    #[inline]
    pub fn as_scale_min(&self) -> &GUPairQ4KScaleMin {
        // SAFETY: `GUPairQ4KScaleMin` is exactly the four trailing arrays of
        // this repr(C, align(16)) struct, and `pair` is 288 bytes (16B aligned).
        unsafe { &*((&self.gate_sc as *const [u8; 8]) as *const GUPairQ4KScaleMin) }
    }
}

/// Pre-unpacked 6-bit scale/min arrays for one gate/up Q4_K pair.
///
/// The scale-plane layout stores this separately so the hot `GUPairQ4K` row stream can
/// keep the regular 288-byte unit while still avoiding per-call scale unpack.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GUPairQ4KScaleMin {
    pub gate_sc: [u8; 8],
    pub gate_m: [u8; 8],
    pub up_sc: [u8; 8],
    pub up_m: [u8; 8],
}

impl GUPairQ4KScaleMin {
    #[inline]
    pub fn from_pair(pair: GUPairQ4K) -> Self {
        let (gate_sc, gate_m) = unpack_k4_scales(&pair.gate.sub_scales);
        let (up_sc, up_m) = unpack_k4_scales(&pair.up.sub_scales);
        Self {
            gate_sc,
            gate_m,
            up_sc,
            up_m,
        }
    }
}

/// Shared expert gate/up block-interleaved unit (Q8_0).
/// 8 Q8_0 blocks cover 256 elems = 1 Q8K activation block.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct SharedGUQ8KUnit {
    pub gate_q8_0: [Q80IntScale; 8],
    pub up_q8_0: [Q80IntScale; 8],
}

#[inline(always)]
fn unpack_k4_scales(q: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let sc = [
        q[0] & 63,
        q[1] & 63,
        q[2] & 63,
        q[3] & 63,
        (q[8] & 0x0F) | ((q[0] >> 6) << 4),
        (q[9] & 0x0F) | ((q[1] >> 6) << 4),
        (q[10] & 0x0F) | ((q[2] >> 6) << 4),
        (q[11] & 0x0F) | ((q[3] >> 6) << 4),
    ];
    let m = [
        q[4] & 63,
        q[5] & 63,
        q[6] & 63,
        q[7] & 63,
        (q[8] >> 4) | ((q[4] >> 6) << 4),
        (q[9] >> 4) | ((q[5] >> 6) << 4),
        (q[10] >> 4) | ((q[6] >> 6) << 4),
        (q[11] >> 4) | ((q[7] >> 6) << 4),
    ];
    (sc, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn q4k_intscale_size() {
        assert_eq!(size_of::<Q4KIntScale>(), 144); // 2 + 128 + padding to 16B
    }

    #[test]
    fn q5k_intscale_size() {
        // 2 + 12 + 128 + 32 = 174, aligned 16 -> 176
        assert_eq!(size_of::<Q5KIntScale>(), 176);
    }

    #[test]
    fn q80_intscale_size() {
        // 2 + 32 = 34, aligned 16 -> 48
        assert_eq!(size_of::<Q80IntScale>(), 48);
    }

    #[test]
    fn gu_pair_size() {
        assert_eq!(size_of::<GUPairQ4K>(), 2 * size_of::<Q4KIntScale>());
    }

    #[test]
    fn gu_pair_unpacked_scales_size() {
        assert_eq!(
            size_of::<GUPairQ4KUnpackedScales>(),
            size_of::<GUPairQ4K>() + 32
        );
    }

    #[test]
    fn gu_pair_unpacked_scales_from_pair_decodes_sub_scales() {
        let mut pair = GUPairQ4K {
            gate: Q4KIntScale {
                scale_int: 3,
                min_int: 5,
                sub_scales: pack_q4k_sub_scales(
                    &[1, 7, 12, 31, 33, 42, 55, 63],
                    &[2, 9, 14, 29, 34, 43, 56, 62],
                ),
                qs: [0u8; 128],
            },
            up: Q4KIntScale {
                scale_int: 7,
                min_int: 11,
                sub_scales: pack_q4k_sub_scales(
                    &[3, 8, 13, 30, 35, 44, 57, 61],
                    &[4, 10, 15, 28, 36, 45, 58, 60],
                ),
                qs: [0u8; 128],
            },
        };
        pair.gate.qs[0] = 17;
        pair.up.qs[0] = 34;

        let unpacked = GUPairQ4KUnpackedScales::from_pair(pair);

        assert_eq!(unpacked.pair.gate.qs[0], 17);
        assert_eq!(unpacked.pair.up.qs[0], 34);
        assert_eq!(unpacked.gate_sc, [1, 7, 12, 31, 33, 42, 55, 63]);
        assert_eq!(unpacked.gate_m, [2, 9, 14, 29, 34, 43, 56, 62]);
        assert_eq!(unpacked.up_sc, [3, 8, 13, 30, 35, 44, 57, 61]);
        assert_eq!(unpacked.up_m, [4, 10, 15, 28, 36, 45, 58, 60]);
    }

    #[test]
    fn gu_pair_scale_min_sidecar_from_pair_is_32_bytes() {
        let pair = GUPairQ4K {
            gate: Q4KIntScale {
                scale_int: 3,
                min_int: 5,
                sub_scales: pack_q4k_sub_scales(
                    &[1, 7, 12, 31, 33, 42, 55, 63],
                    &[2, 9, 14, 29, 34, 43, 56, 62],
                ),
                qs: [0u8; 128],
            },
            up: Q4KIntScale {
                scale_int: 7,
                min_int: 11,
                sub_scales: pack_q4k_sub_scales(
                    &[3, 8, 13, 30, 35, 44, 57, 61],
                    &[4, 10, 15, 28, 36, 45, 58, 60],
                ),
                qs: [0u8; 128],
            },
        };

        let side = GUPairQ4KScaleMin::from_pair(pair);

        assert_eq!(size_of::<GUPairQ4KScaleMin>(), 32);
        assert_eq!(side.gate_sc, [1, 7, 12, 31, 33, 42, 55, 63]);
        assert_eq!(side.gate_m, [2, 9, 14, 29, 34, 43, 56, 62]);
        assert_eq!(side.up_sc, [3, 8, 13, 30, 35, 44, 57, 61]);
        assert_eq!(side.up_m, [4, 10, 15, 28, 36, 45, 58, 60]);
    }

    #[test]
    fn shared_gu_unit_size() {
        assert_eq!(size_of::<SharedGUQ8KUnit>(), 16 * size_of::<Q80IntScale>());
    }

    #[test]
    fn alignment_16() {
        assert_eq!(std::mem::align_of::<Q4KIntScale>(), 16);
        assert_eq!(std::mem::align_of::<Q5KIntScale>(), 16);
        assert_eq!(std::mem::align_of::<Q80IntScale>(), 16);
        assert_eq!(std::mem::align_of::<GUPairQ4K>(), 16);
        assert_eq!(std::mem::align_of::<GUPairQ4KUnpackedScales>(), 16);
        assert_eq!(std::mem::align_of::<GUPairQ4KScaleMin>(), 16);
        assert_eq!(std::mem::align_of::<SharedGUQ8KUnit>(), 16);
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
}
