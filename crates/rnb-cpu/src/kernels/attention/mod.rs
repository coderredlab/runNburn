//! CPU attention kernels.
//!
//! mc74 split layout:
//! - `neon_helpers` (aarch64) — NEON SIMD primitives, including mc72 native fp16
//! - `x86_helpers` (x86/x86_64) — AVX/F16C primitives for KV f32↔f16 dispatch
//! - `dispatch` — cross-arch wrappers picking NEON/AVX/scalar at runtime
//! - `batch_f32` — legacy fp16-fallback batch FlashAttention (default path)
//! - `batch_f16` (aarch64) — mc73 native fp16 batch FlashAttention
//! - `decode_flash` — single-token decode path with mc72 `process_head_f16_acc`

mod dispatch;
#[cfg(target_arch = "aarch64")]
mod neon_helpers;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_helpers;

mod batch_f32;
pub use batch_f32::*;
#[cfg(target_arch = "aarch64")]
mod batch_f16;
#[cfg(target_arch = "aarch64")]
pub use batch_f16::*;
mod decode_flash;
pub use decode_flash::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tensor_as_f32_slice;
    use rnb_core::tensor::Tensor;

    #[test]
    fn test_attention_decode_into_matches_tensor() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let kv_len = 3;
        let kv_dim = num_kv_heads * head_dim;

        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let k_f32: Vec<f32> = (0..kv_len * kv_dim)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v_f32: Vec<f32> = (0..kv_len * kv_dim)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();

        // Convert K/V to F16
        let k_f16: Vec<u16> = k_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let v_f16: Vec<u16> = v_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        // Reference: F32 prefill attention (unchanged)
        let q_t = Tensor::from_slice(&q, &[1, num_heads * head_dim]);
        let k_t = Tensor::from_slice(&k_f32, &[kv_len, kv_dim]);
        let v_t = Tensor::from_slice(&v_f32, &[kv_len, kv_dim]);
        let expected_t = attention(&q_t, &k_t, &v_t, num_heads, num_kv_heads, head_dim).unwrap();
        let expected = tensor_as_f32_slice(&expected_t);

        let mut output = vec![0.0f32; num_heads * head_dim];
        attention_decode_into(
            &q,
            &k_f16,
            &v_f16,
            &mut output,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        // Relaxed tolerance for F16 round-trip
        for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "F16 attention mismatch at {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    fn make_qkv(
        seq: usize,
        kv: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        val: f32,
    ) -> (Tensor, Tensor, Tensor) {
        let q = Tensor::from_slice(&vec![val; seq * nh * hd], &[seq, nh * hd]);
        let k = Tensor::from_slice(&vec![val; kv * nkv * hd], &[kv, nkv * hd]);
        let v = Tensor::from_slice(&vec![val; kv * nkv * hd], &[kv, nkv * hd]);
        (q, k, v)
    }

    #[test]
    fn test_attention_shape() {
        let (q, k, v) = make_qkv(2, 3, 4, 4, 8, 0.1);
        let out = attention(&q, &k, &v, 4, 4, 8).unwrap();
        assert_eq!(out.shape(), &[2, 32]); // [seq_len=2, num_heads*head_dim=32]
    }

    #[test]
    fn test_attention_single_token() {
        // seq_len=1, kv_len=1: 결과가 v와 동일해야 함 (softmax가 1.0 * v)
        let q = Tensor::from_slice(&[1.0f32, 0.0], &[1, 2]);
        let k = Tensor::from_slice(&[1.0f32, 0.0], &[1, 2]);
        let v = Tensor::from_slice(&[3.0f32, 7.0], &[1, 2]);
        let out = attention(&q, &k, &v, 1, 1, 2).unwrap();
        let bytes = out.as_bytes().unwrap();
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // softmax([score]) = [1.0], 결과 = 1.0 * v
        assert!((data[0] - 3.0).abs() < 1e-4, "data[0]={}", data[0]);
        assert!((data[1] - 7.0).abs() < 1e-4, "data[1]={}", data[1]);
    }

    #[test]
    fn test_attention_gqa() {
        // num_heads=4, num_kv_heads=2: GQA
        let (q, k, v) = make_qkv(1, 2, 4, 2, 4, 0.5);
        let out = attention(&q, &k, &v, 4, 2, 4).unwrap();
        assert_eq!(out.shape(), &[1, 16]); // [1, 4*4]
    }

    #[test]
    fn test_attention_invalid_gqa() {
        // num_heads=3, num_kv_heads=2: 3%2 != 0 → 에러
        let (q, k, v) = make_qkv(1, 1, 3, 2, 4, 0.0);
        let result = attention(&q, &k, &v, 3, 2, 4);
        assert!(result.is_err());
    }

    /// Naive reference attention for numerical accuracy testing
    fn naive_attention(
        q_data: &[f32],
        k_data: &[f32],
        v_data: &[f32],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let heads_per_group = num_heads / num_kv_heads;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];

        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            for i in 0..seq_len {
                let q_off = i * num_heads * head_dim + h * head_dim;
                let q_row = &q_data[q_off..q_off + head_dim];
                let global_pos = (kv_len - seq_len) + i;

                let mut scores = vec![f32::NEG_INFINITY; kv_len];
                for j in 0..=global_pos.min(kv_len - 1) {
                    let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                    let k_row = &k_data[k_off..k_off + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_row[d] * k_row[d];
                    }
                    scores[j] = dot * scale;
                }

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                if sum_exp > 0.0 {
                    for s in &mut scores {
                        *s /= sum_exp;
                    }
                }

                let out_off = i * num_heads * head_dim + h * head_dim;
                for j in 0..kv_len {
                    let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                    let v_row = &v_data[v_off..v_off + head_dim];
                    for d in 0..head_dim {
                        out[out_off + d] += scores[j] * v_row[d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_attention_long_seq_matches_naive() {
        // 128-token sequence, 4 heads, 4 kv heads, head_dim=64
        let seq_len = 128;
        let kv_len = 128;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 64;

        // Deterministic pseudo-random data using LCG (seed 42)
        let mut rng_state: u64 = 42;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-1, 1]
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };

        let q_len = seq_len * num_heads * head_dim;
        let kv_data_len = kv_len * num_kv_heads * head_dim;

        let q_vec: Vec<f32> = (0..q_len).map(|_| next_f32()).collect();
        let k_vec: Vec<f32> = (0..kv_data_len).map(|_| next_f32()).collect();
        let v_vec: Vec<f32> = (0..kv_data_len).map(|_| next_f32()).collect();

        let q = Tensor::from_slice(&q_vec, &[seq_len, num_heads * head_dim]);
        let k = Tensor::from_slice(&k_vec, &[kv_len, num_kv_heads * head_dim]);
        let v = Tensor::from_slice(&v_vec, &[kv_len, num_kv_heads * head_dim]);

        // FlashAttention result
        let flash_out = attention(&q, &k, &v, num_heads, num_kv_heads, head_dim).unwrap();
        let flash_bytes = flash_out.as_bytes().unwrap();
        let flash_data: Vec<f32> = flash_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Naive reference result
        let naive_data = naive_attention(
            &q_vec,
            &k_vec,
            &v_vec,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        assert_eq!(flash_data.len(), naive_data.len());

        let mut max_diff = 0.0f32;
        for (i, (f, n)) in flash_data.iter().zip(naive_data.iter()).enumerate() {
            let diff = (f - n).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-4,
                "Mismatch at index {i}: flash={f}, naive={n}, diff={diff}"
            );
        }
        println!("Long-seq test passed. Max diff: {max_diff}");
    }

    #[test]
    fn test_attention_decode_with_window_matches_prefill_last_token() {
        let seq_len = 9;
        let kv_len = 9;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let sliding_window = Some(4usize);
        let scale = 1.0f32;

        let mut rng_state: u64 = 7;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };

        let q_len = seq_len * num_heads * head_dim;
        let kv_data_len = kv_len * num_kv_heads * head_dim;
        let q_vec: Vec<f32> = (0..q_len).map(|_| next_f32()).collect();
        let k_vec: Vec<f32> = (0..kv_data_len).map(|_| next_f32()).collect();
        let v_vec: Vec<f32> = (0..kv_data_len).map(|_| next_f32()).collect();

        let q = Tensor::from_slice(&q_vec, &[seq_len, num_heads * head_dim]);
        let k = Tensor::from_slice(&k_vec, &[kv_len, num_kv_heads * head_dim]);
        let v = Tensor::from_slice(&v_vec, &[kv_len, num_kv_heads * head_dim]);
        let prefill = attention_with_scale_and_window(
            &q,
            &k,
            &v,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            sliding_window,
        )
        .unwrap();
        let prefill_data = tensor_as_f32_slice(&prefill);
        let last_prefill =
            &prefill_data[(seq_len - 1) * num_heads * head_dim..seq_len * num_heads * head_dim];

        let q_last = &q_vec[(seq_len - 1) * num_heads * head_dim..seq_len * num_heads * head_dim];
        let k_f16: Vec<u16> = k_vec
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let v_f16: Vec<u16> = v_vec
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let mut decode = vec![0.0f32; num_heads * head_dim];
        attention_decode_into_with_scale_and_window(
            q_last,
            &k_f16,
            &v_f16,
            &mut decode,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            sliding_window,
        );

        for (i, (a, b)) in decode.iter().zip(last_prefill.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "windowed decode mismatch at {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    /// Flash decode 와 기존 decode 가 cos ≥ 0.999 로 일치하는지 검증
    #[test]
    fn test_attention_decode_flash_matches_legacy() {
        let num_heads = 8;
        let num_kv_heads = 4; // GQA 2:1
        let head_dim = 64;
        let kv_len = 128;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut rng_state: u64 = 12345;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };

        let q_vec: Vec<f32> = (0..num_heads * head_dim).map(|_| next_f32()).collect();
        let k_f32: Vec<f32> = (0..kv_len * kv_dim).map(|_| next_f32()).collect();
        let v_f32: Vec<f32> = (0..kv_len * kv_dim).map(|_| next_f32()).collect();
        let k_f16: Vec<u16> = k_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let v_f16: Vec<u16> = v_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        let mut legacy = vec![0.0f32; num_heads * head_dim];
        attention_decode_into_with_scale(
            &q_vec,
            &k_f16,
            &v_f16,
            &mut legacy,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
        );

        let mut flash = vec![0.0f32; num_heads * head_dim];
        attention_decode_flash(
            &q_vec,
            &k_f16,
            &v_f16,
            &mut flash,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            None,
            None,
        );

        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&a, &b) in legacy.iter().zip(flash.iter()) {
            dot += a as f64 * b as f64;
            norm_a += a as f64 * a as f64;
            norm_b += b as f64 * b as f64;
        }
        let cos = dot / (norm_a.sqrt() * norm_b.sqrt());
        assert!(
            cos >= 0.9999,
            "Flash decode cos similarity too low: {cos} (legacy vs flash)"
        );

        // Also bound max abs diff — online softmax should be bit-close to 2-pass
        let max_abs = legacy
            .iter()
            .zip(flash.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "Flash decode max abs diff too large: {max_abs}"
        );
    }

    /// Flash decode 가 sliding window 와 softcap 을 올바르게 처리
    #[test]
    fn test_attention_decode_flash_with_window_and_softcap() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32;
        let kv_len = 64;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let window = Some(16usize);
        let softcap = Some(50.0f32);

        let mut rng_state: u64 = 999;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };

        let q_vec: Vec<f32> = (0..num_heads * head_dim).map(|_| next_f32()).collect();
        let k_f32: Vec<f32> = (0..kv_len * kv_dim).map(|_| next_f32()).collect();
        let v_f32: Vec<f32> = (0..kv_len * kv_dim).map(|_| next_f32()).collect();
        let k_f16: Vec<u16> = k_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let v_f16: Vec<u16> = v_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        let mut legacy = vec![0.0f32; num_heads * head_dim];
        attention_decode_into_with_scale_window_and_softcap(
            &q_vec,
            &k_f16,
            &v_f16,
            &mut legacy,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            window,
            softcap,
        );

        let mut flash = vec![0.0f32; num_heads * head_dim];
        attention_decode_flash(
            &q_vec,
            &k_f16,
            &v_f16,
            &mut flash,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            window,
            softcap,
        );

        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&a, &b) in legacy.iter().zip(flash.iter()) {
            dot += a as f64 * b as f64;
            norm_a += a as f64 * a as f64;
            norm_b += b as f64 * b as f64;
        }
        let cos = dot / (norm_a.sqrt() * norm_b.sqrt());
        assert!(
            cos >= 0.9999,
            "Flash decode (window+softcap) cos too low: {cos}"
        );
    }
}
