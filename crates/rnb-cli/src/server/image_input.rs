use super::http::ApiError;
use super::responses::unsupported;
use base64::prelude::{Engine as _, BASE64_STANDARD};
use image::{ImageFormat, ImageReader, Limits};
use rnb_llm::Qwen36RgbImage;
use std::io::Cursor;

pub(super) fn decode_data_image(
    url: &str,
    parameter: &'static str,
    field: &str,
) -> Result<Qwen36RgbImage, ApiError> {
    const MAX_ENCODED_BYTES: usize = 28 * 1024 * 1024;
    const MAX_DECODED_ALLOC: u64 = 256 * 1024 * 1024;
    const MAX_DIMENSION: u32 = 8192;

    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(unsupported(
            parameter,
            "remote image URLs are not supported; use a PNG or JPEG data URL",
        ));
    }
    let (header, payload) = url.split_once(',').ok_or_else(|| {
        ApiError::invalid(
            format!("{field} must be a PNG or JPEG data URL"),
            Some(parameter),
            Some("invalid_image"),
        )
    })?;
    let claimed_format = match header {
        "data:image/png;base64" => ImageFormat::Png,
        "data:image/jpeg;base64" => ImageFormat::Jpeg,
        _ => {
            return Err(ApiError::invalid(
                format!("{field} must use data:image/png;base64 or data:image/jpeg;base64"),
                Some(parameter),
                Some("invalid_image"),
            ));
        }
    };
    if payload.len() > MAX_ENCODED_BYTES {
        return Err(ApiError::invalid(
            "image data URL exceeds the 28 MiB encoded limit",
            Some(parameter),
            Some("invalid_image"),
        ));
    }
    let bytes = BASE64_STANDARD.decode(payload).map_err(|error| {
        ApiError::invalid(
            format!("image data URL contains invalid base64: {error}"),
            Some(parameter),
            Some("invalid_image"),
        )
    })?;
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| {
            ApiError::invalid(
                format!("could not identify image data: {error}"),
                Some(parameter),
                Some("invalid_image"),
            )
        })?;
    if reader.format() != Some(claimed_format) {
        return Err(ApiError::invalid(
            "image MIME type does not match the encoded image",
            Some(parameter),
            Some("invalid_image"),
        ));
    }
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_DIMENSION);
    limits.max_image_height = Some(MAX_DIMENSION);
    limits.max_alloc = Some(MAX_DECODED_ALLOC);
    reader.limits(limits);
    let decoded = reader.decode().map_err(|error| {
        ApiError::invalid(
            format!("could not decode image: {error}"),
            Some(parameter),
            Some("invalid_image"),
        )
    })?;
    let rgb = decoded.into_rgb8();
    Qwen36RgbImage::new(rgb.width() as usize, rgb.height() as usize, rgb.into_raw()).map_err(
        |error| {
            ApiError::invalid(
                format!("invalid image: {error}"),
                Some(parameter),
                Some("invalid_image"),
            )
        },
    )
}
