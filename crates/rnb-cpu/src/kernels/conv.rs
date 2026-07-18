use rnb_core::error::Result;
use rnb_core::tensor::Tensor;

use super::tensor_as_f32_slice;

/// Causal depthwise 1D convolution for SSM.
///
/// input: [(kernel_size-1) + seq_len, channels] — conv states prepended
/// kernel: [kernel_size, channels] — depthwise (each channel independent)
/// Returns: [seq_len, channels]
pub fn ssm_conv1d(input: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    let input_data = tensor_as_f32_slice(input);
    let kernel_data = tensor_as_f32_slice(kernel);
    let in_shape = input.shape();
    let k_shape = kernel.shape();

    let total_len = in_shape[0];
    let channels = in_shape[1];
    let kernel_size = k_shape[0];
    let seq_len = total_len - (kernel_size - 1);

    let mut out = vec![0.0f32; seq_len * channels];

    #[cfg(target_arch = "aarch64")]
    {
        ssm_conv1d_neon(
            input_data,
            kernel_data,
            &mut out,
            seq_len,
            channels,
            kernel_size,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for t in 0..seq_len {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for k in 0..kernel_size {
                    sum += input_data[(t + k) * channels + c] * kernel_data[k * channels + c];
                }
                out[t * channels + c] = sum;
            }
        }
    }

    Ok(Tensor::from_vec(out, &[seq_len, channels]))
}

/// Fused conv1d + SiLU: computes silu(conv1d(input, kernel)) in one pass.
/// Saves one allocation and one data traversal.
pub fn ssm_conv1d_silu(input: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    let input_data = tensor_as_f32_slice(input);
    let kernel_data = tensor_as_f32_slice(kernel);
    let in_shape = input.shape();
    let k_shape = kernel.shape();

    let total_len = in_shape[0];
    let channels = in_shape[1];
    let kernel_size = k_shape[0];
    let seq_len = total_len - (kernel_size - 1);

    let mut out = vec![0.0f32; seq_len * channels];

    #[cfg(target_arch = "aarch64")]
    {
        ssm_conv1d_silu_neon(
            input_data,
            kernel_data,
            &mut out,
            seq_len,
            channels,
            kernel_size,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for t in 0..seq_len {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for k in 0..kernel_size {
                    sum += input_data[(t + k) * channels + c] * kernel_data[k * channels + c];
                }
                out[t * channels + c] = sum / (1.0 + (-sum).exp());
            }
        }
    }

    Ok(Tensor::from_vec(out, &[seq_len, channels]))
}

/// Fused conv1d + SiLU, writes to pre-allocated output.
/// input: [(kernel_size-1)+seq_len, channels] flat, output: [seq_len * channels]
pub fn ssm_conv1d_silu_into(
    input_data: &[f32],
    kernel_data: &[f32],
    output: &mut [f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        ssm_conv1d_silu_neon(
            input_data,
            kernel_data,
            output,
            seq_len,
            channels,
            kernel_size,
        );
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for t in 0..seq_len {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for k in 0..kernel_size {
                    sum += input_data[(t + k) * channels + c] * kernel_data[k * channels + c];
                }
                output[t * channels + c] = sum / (1.0 + (-sum).exp());
            }
        }
    }
}

/// NEON-vectorized depthwise conv1d: processes 4 channels at a time.
#[cfg(target_arch = "aarch64")]
fn ssm_conv1d_neon(
    input: &[f32],
    kernel: &[f32],
    out: &mut [f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) {
    use std::arch::aarch64::*;
    let channels_4 = channels / 4 * 4;

    for t in 0..seq_len {
        let out_off = t * channels;
        let mut c = 0;
        while c < channels_4 {
            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                for k in 0..kernel_size {
                    let inp = vld1q_f32(input.as_ptr().add((t + k) * channels + c));
                    let ker = vld1q_f32(kernel.as_ptr().add(k * channels + c));
                    acc = vfmaq_f32(acc, inp, ker);
                }
                vst1q_f32(out.as_mut_ptr().add(out_off + c), acc);
            }
            c += 4;
        }
        // Scalar tail
        while c < channels {
            let mut sum = 0.0f32;
            for k in 0..kernel_size {
                sum += input[(t + k) * channels + c] * kernel[k * channels + c];
            }
            out[out_off + c] = sum;
            c += 1;
        }
    }
}

/// NEON fused conv1d + SiLU: conv then apply x/(1+exp(-x)) in-place.
#[cfg(target_arch = "aarch64")]
fn ssm_conv1d_silu_neon(
    input: &[f32],
    kernel: &[f32],
    out: &mut [f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) {
    use std::arch::aarch64::*;
    let channels_4 = channels / 4 * 4;
    let ones = unsafe { vdupq_n_f32(1.0) };

    for t in 0..seq_len {
        let out_off = t * channels;
        let mut c = 0;
        while c < channels_4 {
            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                for k in 0..kernel_size {
                    let inp = vld1q_f32(input.as_ptr().add((t + k) * channels + c));
                    let ker = vld1q_f32(kernel.as_ptr().add(k * channels + c));
                    acc = vfmaq_f32(acc, inp, ker);
                }
                // SiLU: x / (1 + exp(-x)) — no NEON exp, use scalar per lane
                let mut buf = [0.0f32; 4];
                vst1q_f32(buf.as_mut_ptr(), acc);
                buf[0] = buf[0] / (1.0 + (-buf[0]).exp());
                buf[1] = buf[1] / (1.0 + (-buf[1]).exp());
                buf[2] = buf[2] / (1.0 + (-buf[2]).exp());
                buf[3] = buf[3] / (1.0 + (-buf[3]).exp());
                vst1q_f32(out.as_mut_ptr().add(out_off + c), vld1q_f32(buf.as_ptr()));
            }
            c += 4;
        }
        while c < channels {
            let mut sum = 0.0f32;
            for k in 0..kernel_size {
                sum += input[(t + k) * channels + c] * kernel[k * channels + c];
            }
            out[out_off + c] = sum / (1.0 + (-sum).exp());
            c += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tensor_to_f32_vec;

    #[test]
    fn test_conv1d_identity() {
        // kernel=[1,0,0] with 3 channels, identity-like
        // input: [3+2, 3] = [5, 3] (kernel_size-1=2 states + 3 tokens)
        let input = Tensor::from_slice(
            &[
                // conv states (2 rows)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // actual tokens (3 rows)
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            ],
            &[5, 3],
        );
        // kernel [3, 3]: last position = 1.0, others = 0.0
        let kernel = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 3]);
        let out = ssm_conv1d(&input, &kernel).unwrap();
        let data = tensor_to_f32_vec(&out);
        assert_eq!(out.shape(), &[3, 3]);
        // Should pick last element of each window = the current token
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv1d_sum() {
        // kernel=[1,1] with 1 channel, sums adjacent
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3, 1]); // 1 state + 2 tokens
        let kernel = Tensor::from_slice(&[1.0, 1.0], &[2, 1]);
        let out = ssm_conv1d(&input, &kernel).unwrap();
        let data = tensor_to_f32_vec(&out);
        assert_eq!(out.shape(), &[2, 1]);
        assert!((data[0] - 3.0).abs() < 1e-5); // 1+2
        assert!((data[1] - 5.0).abs() < 1e-5); // 2+3
    }
}
