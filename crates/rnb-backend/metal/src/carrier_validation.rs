#[inline]
pub(crate) fn assert_exact_len(label: &str, actual: usize, expected: usize) {
    assert_eq!(
        actual, expected,
        "Metal carrier {label} length mismatch: got {actual}, expected {expected}"
    );
}

#[inline]
pub(crate) fn checked_product(label: &str, left: usize, right: usize) -> usize {
    left.checked_mul(right)
        .unwrap_or_else(|| panic!("Metal carrier {label} length overflow: {left} * {right}"))
}

#[inline]
pub(crate) fn checked_slot_range_end(
    label: &str,
    start: usize,
    count: usize,
    capacity: usize,
) -> usize {
    let end = start
        .checked_add(count)
        .unwrap_or_else(|| panic!("Metal carrier {label} slot range overflow: {start} + {count}"));
    assert!(
        end <= capacity,
        "Metal carrier {label} slot range exceeds capacity: end {end}, capacity {capacity}"
    );
    assert!(
        end <= u32::MAX as usize,
        "Metal carrier {label} slot range exceeds u32: end {end}"
    );
    end
}

#[cfg(test)]
mod tests {
    use super::{assert_exact_len, checked_product, checked_slot_range_end};

    #[test]
    fn accepts_exact_length() {
        assert_exact_len("hidden", 4, 4);
    }

    #[test]
    #[should_panic(expected = "Metal carrier hidden length mismatch")]
    fn rejects_mismatched_length_in_release() {
        assert_exact_len("hidden", 3, 4);
    }

    #[test]
    fn accepts_bounded_slot_range() {
        assert_eq!(checked_slot_range_end("KV", 3, 2, 8), 5);
    }

    #[test]
    #[should_panic(expected = "slot range exceeds capacity")]
    fn rejects_slot_range_past_capacity_in_release() {
        checked_slot_range_end("KV", 7, 2, 8);
    }

    #[test]
    #[should_panic(expected = "slot range overflow")]
    fn rejects_slot_range_overflow_in_release() {
        checked_slot_range_end("KV", usize::MAX, 1, usize::MAX);
    }

    #[test]
    #[should_panic(expected = "length overflow")]
    fn rejects_length_product_overflow_in_release() {
        checked_product("KV", usize::MAX, 2);
    }
}
