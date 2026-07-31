pub(crate) fn page_align(
    ptr: usize,
    raw_len: usize,
    page_size: usize,
) -> Result<(usize, usize, usize), &'static str> {
    if !page_size.is_power_of_two() {
        return Err("page size must be a non-zero power of two");
    }

    let end = ptr
        .checked_add(raw_len)
        .ok_or("NoCopy pointer range overflow")?;
    let aligned = ptr & !(page_size - 1);
    let page_offset = ptr
        .checked_sub(aligned)
        .ok_or("NoCopy page offset underflow")?;
    let span = end
        .checked_sub(aligned)
        .ok_or("NoCopy aligned span underflow")?;
    let round_up = page_size
        .checked_sub(1)
        .ok_or("page size must be non-zero")?;
    let buf_len = span
        .checked_add(round_up)
        .ok_or("NoCopy rounded range overflow")?
        .checked_div(page_size)
        .and_then(|pages| pages.checked_mul(page_size))
        .ok_or("NoCopy rounded range overflow")?;
    aligned
        .checked_add(buf_len)
        .ok_or("NoCopy rounded range overflow")?;
    Ok((aligned, page_offset, buf_len))
}

#[cfg(test)]
mod tests {
    use super::page_align;

    #[test]
    fn aligns_for_4k_and_16k_host_pages() {
        let ptr = 0x5500;
        let raw_len = 10_000;

        assert_eq!(page_align(ptr, raw_len, 4096), Ok((0x5000, 0x500, 12_288)));
        assert_eq!(
            page_align(ptr, raw_len, 16_384),
            Ok((0x4000, 0x1500, 16_384))
        );
    }

    #[test]
    fn exact_page_multiple_len_is_not_extended() {
        assert_eq!(page_align(16_384, 16_384, 16_384), Ok((16_384, 0, 16_384)));
    }

    #[test]
    fn rejects_invalid_page_sizes() {
        assert_eq!(
            page_align(4096, 100, 0),
            Err("page size must be a non-zero power of two")
        );
        assert_eq!(
            page_align(4096, 100, 6000),
            Err("page size must be a non-zero power of two")
        );
    }

    #[test]
    fn rejects_pointer_range_overflow() {
        assert_eq!(
            page_align(usize::MAX, 1, 4096),
            Err("NoCopy pointer range overflow")
        );
    }

    #[test]
    fn rejects_round_up_overflow() {
        assert_eq!(
            page_align(1, usize::MAX - 1, 4096),
            Err("NoCopy rounded range overflow")
        );
        assert_eq!(
            page_align(usize::MAX, 0, 4096),
            Err("NoCopy rounded range overflow")
        );
    }
}
