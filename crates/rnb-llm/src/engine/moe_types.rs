/// Constant byte/block counts for Q4_K and Q5_1 rows in our expert tensors.
#[inline]
pub fn q4k_bytes_per_row(cols: usize) -> usize {
    debug_assert!(cols % 256 == 0, "Q4_K requires cols divisible by 256");
    (cols / 256) * 144
}

#[inline]
pub fn q5_1_bytes_per_row(cols: usize) -> usize {
    debug_assert!(cols % 32 == 0, "Q5_1 requires cols divisible by 32");
    (cols / 32) * 24
}

/// Session 71 MoE mixed precision: Q2_K row bytes (84 B per 256 elements).
#[inline]
pub fn q2k_bytes_per_row(cols: usize) -> usize {
    debug_assert!(cols % 256 == 0, "Q2_K requires cols divisible by 256");
    (cols / 256) * 84
}

#[inline]
pub fn q3k_bytes_per_row(cols: usize) -> usize {
    debug_assert!(cols % 256 == 0, "Q3_K requires cols divisible by 256");
    (cols / 256) * 110
}

#[inline]
pub fn q5k_bytes_per_row(cols: usize) -> usize {
    debug_assert!(cols % 256 == 0, "Q5_K requires cols divisible by 256");
    (cols / 256) * 176
}

#[inline]
pub fn q6k_bytes_per_row(cols: usize) -> usize {
    debug_assert!(cols % 256 == 0, "Q6_K requires cols divisible by 256");
    (cols / 256) * 210
}

#[inline]
pub fn iq4_xs_bytes_per_row(cols: usize) -> usize {
    debug_assert!(cols % 256 == 0, "IQ4_XS requires cols divisible by 256");
    (cols / 256) * 136
}

#[inline]
pub fn q8_0_bytes_per_row(cols: usize) -> usize {
    debug_assert!(cols % 32 == 0, "Q8_0 requires cols divisible by 32");
    (cols / 32) * 34
}
