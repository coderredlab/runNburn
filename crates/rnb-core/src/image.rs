use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError("image dimensions must be positive".into()));
        }
        if width > u32::MAX as usize || height > u32::MAX as usize {
            return Err(ImageError("image dimensions must fit u32".into()));
        }
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| ImageError("RGB image byte count overflows usize".into()))?;
        if pixels.len() != expected {
            return Err(ImageError(format!(
                "RGB image has {} bytes, expected {expected} for {width}x{height}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageError(String);

impl fmt::Display for ImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ImageError {}

#[cfg(test)]
mod tests {
    use super::RgbImage;

    #[test]
    fn rejects_invalid_dimensions_and_storage() {
        assert!(RgbImage::new(0, 1, Vec::new()).is_err());
        assert!(RgbImage::new(2, 2, vec![0; 11]).is_err());
    }
}
