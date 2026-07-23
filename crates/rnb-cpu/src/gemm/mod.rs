pub mod activation_q8;
#[cfg(target_arch = "x86_64")]
pub mod avx2_dot;
pub mod dequant;
pub mod f32_gemv;
#[cfg(target_arch = "aarch64")]
pub mod neon_dot;
#[cfg(target_arch = "aarch64")]
pub mod neon_quant;
#[cfg(target_arch = "aarch64")]
pub mod neon_repacked;
pub mod pack_q4k;
pub mod pack_q5k;
pub mod pack_q6k;
pub mod pack_q8_0_pair;
pub mod policy;
pub mod quant_gemv;
pub mod quantized_packing;
pub mod repack;
pub mod tile_q4k;
pub mod tile_q5k;
pub mod tile_q6k;

/// NR (output row tile size) — i8mm smmla 기준
pub const NR: usize = 8;

/// KR (K-dimension block size) — super-block 크기
pub const KR: usize = 256;

#[cfg(target_arch = "aarch64")]
pub use activation_q8::{
    quantize_input_q8, quantize_input_q8_into, quantize_input_q8k, quantize_input_q8k_into,
    Q8Block, Q8KBlock,
};
