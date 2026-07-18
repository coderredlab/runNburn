use std::ops::Range;
use std::sync::Arc;

use crate::error::{Result, RnbError};
use crate::tensor::dtype::{DType, TensorElement};
use crate::tensor::quant::QuantMeta;
use crate::tensor::storage::{Buffer, FileBackedRegion, Storage};

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    dtype: DType,
    quant: Option<QuantMeta>,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl Tensor {
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let size = numel * dtype.size_bytes();
        let buf = Buffer::alloc(size, 64);
        let strides = compute_strides(shape);
        Self {
            storage: Arc::new(Storage::Owned(buf)),
            shape: shape.to_vec(),
            strides,
            offset: 0,
            dtype,
            quant: None,
        }
    }

    pub fn from_slice<T: TensorElement>(data: &[T], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length must match shape");
        let byte_len = numel * std::mem::size_of::<T>();
        let mut buf = Buffer::alloc(byte_len, 64);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.as_mut_ptr(), byte_len);
        }
        let strides = compute_strides(shape);
        Self {
            storage: Arc::new(Storage::Owned(buf)),
            shape: shape.to_vec(),
            strides,
            offset: 0,
            dtype: T::DTYPE,
            quant: None,
        }
    }

    /// Create Tensor from Vec without copying (zero-copy, takes ownership).
    pub fn from_vec<T: TensorElement>(data: Vec<T>, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length must match shape");
        let buf = Buffer::from_vec(data);
        let strides = compute_strides(shape);
        Self {
            storage: Arc::new(Storage::Owned(buf)),
            shape: shape.to_vec(),
            strides,
            offset: 0,
            dtype: T::DTYPE,
            quant: None,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn quant(&self) -> Option<&QuantMeta> {
        self.quant.as_ref()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides(&self.shape);
        self.strides == expected
    }

    pub fn view(&self, new_shape: &[usize]) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(RnbError::ShapeMismatch {
                expected: self.shape.clone(),
                got: new_shape.to_vec(),
            });
        }
        if !self.is_contiguous() {
            return Err(RnbError::InvalidGraph(
                "view requires contiguous tensor".into(),
            ));
        }
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape.to_vec(),
            strides: compute_strides(new_shape),
            offset: self.offset,
            dtype: self.dtype,
            quant: self.quant.clone(),
        })
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            quant: self.quant.clone(),
        }
    }

    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        let numel = self.numel();
        let elem_size = self.dtype.size_bytes();
        let mut buf = Buffer::alloc(numel * elem_size, 64);
        let src = self
            .storage
            .as_slice()
            .expect("contiguous requires host storage");
        let dst = buf.as_mut_slice();
        let ndim = self.shape.len();
        let mut src_indices = vec![0usize; ndim];
        for i in 0..numel {
            let src_offset = self.offset
                + src_indices
                    .iter()
                    .zip(self.strides.iter())
                    .map(|(idx, s)| idx * s)
                    .sum::<usize>();
            let src_byte = src_offset * elem_size;
            let dst_byte = i * elem_size;
            dst[dst_byte..dst_byte + elem_size]
                .copy_from_slice(&src[src_byte..src_byte + elem_size]);
            for d in (0..ndim).rev() {
                src_indices[d] += 1;
                if src_indices[d] < self.shape[d] {
                    break;
                }
                src_indices[d] = 0;
            }
        }
        Self {
            storage: Arc::new(Storage::Owned(buf)),
            shape: self.shape.clone(),
            strides: compute_strides(&self.shape),
            offset: 0,
            dtype: self.dtype,
            quant: self.quant.clone(),
        }
    }

    /// 연속적인 호스트 텐서의 원시 바이트 슬라이스를 반환함.
    /// 텐서가 연속적이고 호스트 메모리에 있을 때만 Some을 반환.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if !self.is_contiguous() {
            return None;
        }
        let raw = self.storage.as_slice()?;
        let start = self.offset * self.dtype.size_bytes();
        let end = start + self.numel() * self.dtype.size_bytes();
        Some(&raw[start..end])
    }

    pub fn file_backed_region(&self) -> Option<FileBackedRegion> {
        if !self.is_contiguous() {
            return None;
        }
        let Storage::FileMmap(storage) = self.storage.as_ref() else {
            return None;
        };
        let bytes = self.as_bytes()?;
        let file_offset = self.offset.checked_mul(self.dtype.size_bytes())? as u64;
        Some(FileBackedRegion::new(
            storage.path_arc(),
            file_offset,
            bytes.len(),
            bytes.as_ptr() as usize,
        ))
    }
    /// Releases clean pages from an immutable file-backed tensor mapping.
    ///
    /// Anonymous mappings and owned/device storage are left unchanged.
    pub fn reclaim_file_mmap_range(
        &self,
        relative_byte_offset: usize,
        byte_len: usize,
    ) -> std::io::Result<()> {
        if byte_len == 0 {
            return Ok(());
        }
        let tensor_byte_len = self
            .numel()
            .checked_mul(self.dtype.size_bytes())
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "tensor size overflow")
            })?;
        let relative_end = relative_byte_offset.checked_add(byte_len).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "tensor range overflow")
        })?;
        if relative_end > tensor_byte_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "tensor reclaim range is out of bounds",
            ));
        }

        #[cfg(unix)]
        if let Storage::FileMmap(mmap) = self.storage.as_ref() {
            let tensor_start = self
                .offset
                .checked_mul(self.dtype.size_bytes())
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "tensor offset overflow")
                })?;
            let mmap_offset = tensor_start
                .checked_add(relative_byte_offset)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "mmap offset overflow")
                })?;
            let mmap_end = mmap_offset.checked_add(byte_len).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "mmap range overflow")
            })?;
            if mmap_end > mmap.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "mmap reclaim range is out of bounds",
                ));
            }
            // SAFETY: `FileMmap` is created only for immutable GGUF file
            // mappings. `MADV_DONTNEED` discards clean file-backed pages and
            // reloads identical bytes on the next access; it cannot discard
            // anonymous or private mutable state.
            unsafe {
                mmap.unchecked_advise_range(
                    memmap2::UncheckedAdvice::DontNeed,
                    mmap_offset,
                    byte_len,
                )
            }?;
        }

        Ok(())
    }

    /// mmap 전체를 Arc<Storage>로 공유받아, byte_offset부터 shape 크기만큼 뷰를 만든다.
    ///
    /// # Arguments
    /// * `storage` - 전체 mmap을 담은 `Arc<Storage::Mmap(...)>` (여러 텐서가 공유)
    /// * `byte_offset` - 파일 내 텐서 데이터의 절대 바이트 위치
    /// * `shape` - 텐서 shape (row-major)
    /// * `dtype` - 텐서 원소 타입
    ///
    /// # Errors
    /// `dtype.size_bytes() == 0` 이거나 `byte_offset`이 원소 크기로 나누어지지 않으면 에러.
    pub fn from_mmap(
        storage: Arc<Storage>,
        byte_offset: usize,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Self> {
        let elem_size = dtype.size_bytes();
        if elem_size == 0 {
            return Err(RnbError::InvalidGraph(
                "dtype has zero element size".to_string(),
            ));
        }
        if !byte_offset.is_multiple_of(elem_size) {
            return Err(RnbError::InvalidGraph(format!(
                "byte_offset {byte_offset} not aligned to element size {elem_size}"
            )));
        }
        let elem_offset = byte_offset / elem_size;
        let strides = compute_strides(shape);
        Ok(Self {
            storage,
            shape: shape.to_vec(),
            strides,
            offset: elem_offset,
            dtype,
            quant: None,
        })
    }

    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<Self> {
        if ranges.len() != self.shape.len() {
            return Err(RnbError::InvalidGraph(format!(
                "slice: expected {} ranges, got {}",
                self.shape.len(),
                ranges.len()
            )));
        }
        let mut new_shape = Vec::with_capacity(ranges.len());
        let mut new_offset = self.offset;
        for (i, range) in ranges.iter().enumerate() {
            if range.end > self.shape[i] {
                return Err(RnbError::InvalidGraph(format!(
                    "slice: range {}..{} out of bounds for dim {} (size {})",
                    range.start, range.end, i, self.shape[i]
                )));
            }
            new_shape.push(range.end - range.start);
            new_offset += range.start * self.strides[i];
        }
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
            dtype: self.dtype,
            quant: self.quant.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.size_bytes(), 24);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_from_slice_f32() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_strides_row_major() {
        let t = Tensor::zeros(&[2, 3, 4], DType::F32);
        assert_eq!(t.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_view_compatible() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        let v = t.view(&[6]).unwrap();
        assert_eq!(v.shape(), &[6]);
        assert_eq!(v.numel(), 6);
    }

    #[test]
    fn test_view_incompatible() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        assert!(t.view(&[5]).is_err());
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        let tr = t.transpose(0, 1);
        assert_eq!(tr.shape(), &[3, 2]);
        assert_eq!(tr.strides(), &[1, 3]);
        assert!(!tr.is_contiguous());
    }

    #[test]
    fn test_contiguous_copy() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        let tr = t.transpose(0, 1);
        assert!(!tr.is_contiguous());
        let c = tr.contiguous();
        assert!(c.is_contiguous());
        assert_eq!(c.shape(), &[3, 2]);
    }

    #[test]
    fn test_slice() {
        let t = Tensor::zeros(&[4, 4], DType::F32);
        let s = t.slice(&[1..3, 0..4]).unwrap();
        assert_eq!(s.shape(), &[2, 4]);
    }
    #[cfg(unix)]
    #[test]
    fn reclaim_file_mmap_range_preserves_file_backed_bytes() {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0x5a; 8192]).unwrap();
        file.flush().unwrap();
        let mmap = unsafe { memmap2::MmapOptions::new().map(file.as_file()).unwrap() };
        let storage = Arc::new(Storage::FileMmap(crate::tensor::FileMmapStorage::new(
            mmap,
            file.path(),
        )));
        let tensor = Tensor::from_mmap(storage, 0, &[8192], DType::U8).unwrap();

        assert_eq!(tensor.as_bytes().unwrap()[4096], 0x5a);
        tensor.reclaim_file_mmap_range(1024, 4096).unwrap();
        assert_eq!(tensor.as_bytes().unwrap()[4096], 0x5a);
        assert!(tensor.reclaim_file_mmap_range(8192, 1).is_err());

        let region = tensor.file_backed_region().unwrap();
        assert_eq!(region.path(), file.path());
        assert_eq!(region.file_offset(), 0);
        assert_eq!(region.len(), 8192);
        assert_eq!(
            region.resolve_subslice(&tensor.as_bytes().unwrap()[1024..2048]),
            Some((1024, 1024))
        );
    }
}
