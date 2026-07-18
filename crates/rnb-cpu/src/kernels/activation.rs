use rayon::prelude::*;
use rnb_core::error::Result;
use rnb_core::tensor::Tensor;

use super::tensor_as_f32_slice;

/// SiLU (Swish): f(x) = x * sigmoid(x) = x / (1 + e^(-x))
pub fn silu(input: &Tensor) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let out: Vec<f32> = x.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    Ok(Tensor::from_vec(out, input.shape()))
}

/// GeLU (근사값): f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let out: Vec<f32> = x
        .iter()
        .map(|&v| 0.5 * v * (1.0 + (sqrt_2_over_pi * (v + 0.044715 * v.powi(3))).tanh()))
        .collect();
    Ok(Tensor::from_vec(out, input.shape()))
}

/// Softmax: exp(x - max) / sum(exp(x - max)), 수치 안정성을 위해 max 빼줌
pub fn softmax(input: &Tensor) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let out: Vec<f32> = exp_vals.iter().map(|&v| v / sum).collect();
    Ok(Tensor::from_vec(out, input.shape()))
}

/// softplus(x) = log(1 + exp(x))
pub fn softplus(input: &Tensor) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let out: Vec<f32> = x
        .iter()
        .map(|&v| {
            if v > 20.0 {
                v
            } else {
                (1.0 + v.exp()).ln()
            } // numerical stability
        })
        .collect();
    Ok(Tensor::from_vec(out, input.shape()))
}

/// sigmoid(x) = 1 / (1 + exp(-x))
pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let out: Vec<f32> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    Ok(Tensor::from_vec(out, input.shape()))
}

/// sigmoid in-place: x[i] = 1 / (1 + exp(-x[i]))
pub fn sigmoid_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

const MIN_PARALLEL_ACTIVATION_ELEMENTS_PER_THREAD: usize = 4096 / std::mem::size_of::<f32>();

#[inline]
fn use_parallel_activation(len: usize) -> bool {
    let threads = rayon::current_num_threads().max(1);
    threads > 1 && len >= threads * MIN_PARALLEL_ACTIVATION_ELEMENTS_PER_THREAD
}

/// Fused SiLU-gate: gate[i] = silu(gate[i]) * up[i], in-place on gate.
pub fn fused_silu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    if use_parallel_activation(gate.len()) {
        gate.par_iter_mut()
            .zip(up.par_iter())
            .for_each(|(gate, up)| {
                let g = *gate;
                *gate = (g / (1.0 + (-g).exp())) * *up;
            });
    } else {
        gate.iter_mut().zip(up.iter()).for_each(|(gate, up)| {
            let g = *gate;
            *gate = (g / (1.0 + (-g).exp())) * *up;
        });
    }
}

pub fn fused_gelu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let activate = |gate: &mut f32, up: &f32| {
        let g = *gate;
        let gelu = 0.5 * g * (1.0 + (sqrt_2_over_pi * (g + 0.044715 * g.powi(3))).tanh());
        *gate = gelu * *up;
    };
    if use_parallel_activation(gate.len()) {
        gate.par_iter_mut()
            .zip(up.par_iter())
            .for_each(|(gate, up)| {
                activate(gate, up);
            });
    } else {
        gate.iter_mut().zip(up.iter()).for_each(|(gate, up)| {
            activate(gate, up);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tensor_to_f32_vec;

    #[test]
    fn test_silu() {
        let input = Tensor::from_slice(&[0.0f32, 1.0], &[1, 2]);
        let output = silu(&input).unwrap();
        let data = tensor_to_f32_vec(&output);
        assert!(data[0].abs() < 1e-5); // silu(0) = 0
        assert!((data[1] - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_gelu() {
        let input = Tensor::from_slice(&[0.0f32, 1.0], &[1, 2]);
        let output = gelu(&input).unwrap();
        let data = tensor_to_f32_vec(&output);
        assert!(data[0].abs() < 1e-5); // gelu(0) = 0
        assert!((data[1] - 0.8412).abs() < 1e-3);
    }

    #[test]
    fn test_softplus() {
        let input = Tensor::from_slice(&[0.0f32, 1.0, -1.0, 10.0], &[4]);
        let output = softplus(&input).unwrap();
        let data = tensor_to_f32_vec(&output);
        assert!((data[0] - std::f32::consts::LN_2).abs() < 1e-3);
        assert!((data[3] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_sigmoid() {
        let input = Tensor::from_slice(&[0.0f32, 100.0, -100.0], &[3]);
        let output = sigmoid(&input).unwrap();
        let data = tensor_to_f32_vec(&output);
        assert!((data[0] - 0.5).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-3);
        assert!(data[2].abs() < 1e-3);
    }

    #[test]
    fn test_softmax() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
        let output = softmax(&input).unwrap();
        let data = tensor_to_f32_vec(&output);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_fused_gelu_mul_inplace() {
        let mut gate = vec![1.0f32, -1.0];
        let up = vec![2.0f32, 3.0];
        fused_gelu_mul_inplace(&mut gate, &up);

        assert!((gate[0] - 1.6824).abs() < 1e-3);
        assert!((gate[1] + 0.4764).abs() < 1e-3);
    }
    #[test]
    fn parallel_fused_activations_match_scalar_oracle() {
        let len = rayon::current_num_threads().max(2) * MIN_PARALLEL_ACTIVATION_ELEMENTS_PER_THREAD;
        let source: Vec<f32> = (0..len)
            .map(|index| (index % 257) as f32 / 32.0 - 4.0)
            .collect();
        let up: Vec<f32> = (0..len)
            .map(|index| (index % 113) as f32 / 64.0 - 0.75)
            .collect();

        let mut silu = source.clone();
        let expected_silu: Vec<f32> = source
            .iter()
            .zip(up.iter())
            .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect();
        fused_silu_mul_inplace(&mut silu, &up);
        assert_eq!(silu, expected_silu);

        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let mut gelu = source.clone();
        let expected_gelu: Vec<f32> = source
            .iter()
            .zip(up.iter())
            .map(|(&gate, &up)| {
                let activated =
                    0.5 * gate * (1.0 + (sqrt_2_over_pi * (gate + 0.044715 * gate.powi(3))).tanh());
                activated * up
            })
            .collect();
        fused_gelu_mul_inplace(&mut gelu, &up);
        assert_eq!(gelu, expected_gelu);
    }
}
