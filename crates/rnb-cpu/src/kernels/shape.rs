use rnb_core::error::{Result, RnbError};
use rnb_core::tensor::Tensor;

use super::tensor_as_f32_slice;

/// 텐서를 새로운 shape으로 reshape. numel은 동일해야 함.
pub fn reshape(input: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
    input.view(new_shape)
}

/// 두 dimension을 transpose.
pub fn transpose_op(input: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor> {
    Ok(input.transpose(dim0, dim1))
}

/// Embedding lookup: table[indices]
/// table: [vocab_size, embed_dim], indices: [seq_len]
pub fn gather(table: &Tensor, indices: &[u32]) -> Result<Tensor> {
    let embed_dim = table.shape().last().copied().unwrap_or(1);
    let table_data = tensor_as_f32_slice(table);
    let mut out = Vec::with_capacity(indices.len() * embed_dim);
    for &idx in indices {
        let start = idx as usize * embed_dim;
        out.extend_from_slice(&table_data[start..start + embed_dim]);
    }
    Ok(Tensor::from_vec(out, &[indices.len(), embed_dim]))
}

/// 여러 텐서를 axis 기준으로 concat.
/// axis 외 나머지 dim은 같아야 함.
pub fn concat(tensors: &[&Tensor], axis: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(RnbError::InvalidGraph("empty concat".into()));
    }
    let mut all_data = Vec::new();
    for t in tensors {
        all_data.extend_from_slice(tensor_as_f32_slice(t));
    }

    let mut new_shape = tensors[0].shape().to_vec();
    let total_on_axis: usize = tensors.iter().map(|t| t.shape()[axis]).sum();
    new_shape[axis] = total_on_axis;

    Ok(Tensor::from_vec(all_data, &new_shape))
}

/// axis 기준으로 텐서를 sizes 크기로 분할.
pub fn split(input: &Tensor, sizes: &[usize], axis: usize) -> Result<Vec<Tensor>> {
    let data = tensor_as_f32_slice(input);
    let mut results = Vec::new();
    let mut offset = 0;
    let axis_size = input.shape()[axis];
    // axis 제외한 나머지 element 수
    let other_elems = input.numel() / axis_size;
    for &size in sizes {
        let elem_count = size * other_elems;
        let chunk = &data[offset..offset + elem_count];
        let mut new_shape = input.shape().to_vec();
        new_shape[axis] = size;
        results.push(Tensor::from_slice(chunk, &new_shape));
        offset += elem_count;
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tensor_to_f32_vec;

    #[test]
    fn test_gather() {
        let table = Tensor::from_slice(
            &[
                0.0f32, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            ],
            &[4, 3],
        );
        let output = gather(&table, &[1, 3]).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
        let data = tensor_to_f32_vec(&output);
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[3] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_concat_axis1() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let b = Tensor::from_slice(&[3.0f32, 4.0, 5.0], &[1, 3]);
        let c = concat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape(), &[1, 5]);
    }

    #[test]
    fn test_split() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
        let parts = split(&t, &[2, 2, 2], 1).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[1, 2]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = reshape(&t, &[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose_op() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let tr = transpose_op(&t, 0, 1).unwrap();
        assert_eq!(tr.shape(), &[2, 2]);
    }
}
