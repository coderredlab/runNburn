//! Per-kv-head shard byte-range computation for `[num_kv_heads * head_dim, hidden]`
//! row-major quantized K/V weights.
//!
//! GGUF stores combined K (or V) projection as one tensor with shape
//! `[num_kv_heads * head_dim, hidden]`, row-major: rows = output channels,
//! cols = input channels. Each row in a Q-quant layout occupies
//! `(hidden / block_size) * bytes_per_block` bytes.
//!
//! For the per-kv-head sharded full-path, we want to upload each kv-head's
//! `head_dim` rows as a separate `WeightKind::KProjShard(kvh)` /
//! `WeightKind::VProjShard(kvh)` cache entry. This module computes the
//! `Range<usize>` that the caller can slice from the combined GGUF blob —
//! zero copy, zero allocation — and pass into `WeightCache::get_or_upload`.
//!
//! Block geometry is sourced from [`QuantType`] so every supported GGUF
//! projection format uses the same row-byte contract.

use core::ops::Range;

use crate::weight_cache::QuantType;

/// Compute the byte length of a single row of a row-major quantized weight tensor
/// with `hidden` columns, for the given `quant`.
///
/// Internal helper for [`kv_head_shard_byte_range`]. Not part of the public API —
/// 4c's caller-side weight upload path slices byte ranges via
/// `kv_head_shard_byte_range` only. Promote back to `pub` if a future caller
/// genuinely needs row-byte computation from outside this crate.
///
/// Returns an error if `hidden == 0` or if `hidden` is not a positive multiple of
/// the quant's block element count (super-block alignment).
///
/// For:
/// - Q4_K / Q5_K / Q6_K → `hidden` must be a positive multiple of 256.
/// - Q8_0 → `hidden` must be a positive multiple of 32.
pub(crate) fn row_bytes_per_row(quant: QuantType, hidden: usize) -> Result<usize, String> {
    if hidden == 0 {
        return Err(format!(
            "row_bytes_per_row: hidden must be > 0 (got 0) for quant {:?}",
            quant
        ));
    }
    let block_elems = quant.block_elements();
    let block_bytes = quant.block_bytes();
    if hidden % block_elems != 0 {
        return Err(format!(
            "row_bytes_per_row: hidden ({}) is not a multiple of block_size ({}) for quant {:?}",
            hidden, block_elems, quant
        ));
    }
    let blocks_per_row = hidden / block_elems;
    Ok(blocks_per_row * block_bytes)
}

/// Compute the byte range within a row-major `[num_kv_heads * head_dim, hidden]`
/// quantized tensor that corresponds to the `kvh`-th kv-head shard.
///
/// The shard covers rows `kvh * head_dim .. (kvh + 1) * head_dim`. Returns
/// `Range { start, end }` in bytes — caller can slice `&combined_bytes[range]`
/// and pass it into `WeightCache::get_or_upload` with
/// `WeightKind::KProjShard(kvh)` / `VProjShard(kvh)`.
///
/// # Caller convention
///
/// `kvh` is `usize` here for ergonomic indexing in caller loops, but
/// `WeightKind::KProjShard(kvh)` / `VProjShard(kvh)` (defined in
/// `weight_cache.rs`) takes `u16`. Cast at the cache call site:
///
/// ```ignore
/// WeightId { kind: WeightKind::KProjShard(kvh as u16), ... }
/// ```
///
/// Realistic `num_kv_heads` stays well below `u16::MAX`, so the cast is
/// lossless.
///
/// # Errors
///
/// - `kvh >= num_kv_heads`
/// - any of `num_kv_heads`, `head_dim`, `hidden` is zero
/// - `hidden` is not a multiple of the quant's block-element count (delegated to
///   [`row_bytes_per_row`])
pub fn kv_head_shard_byte_range(
    num_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    quant: QuantType,
    kvh: usize,
) -> Result<Range<usize>, String> {
    if num_kv_heads == 0 {
        return Err(format!(
            "kv_head_shard_byte_range: num_kv_heads must be > 0 (got 0) for quant {:?}",
            quant
        ));
    }
    if head_dim == 0 {
        return Err(format!(
            "kv_head_shard_byte_range: head_dim must be > 0 (got 0) for quant {:?}",
            quant
        ));
    }
    if kvh >= num_kv_heads {
        return Err(format!(
            "kv_head_shard_byte_range: kvh ({}) is out of range [0, {}) for quant {:?}",
            kvh, num_kv_heads, quant
        ));
    }
    let row_bytes = row_bytes_per_row(quant, hidden)?;
    let shard_bytes = row_bytes.checked_mul(head_dim).ok_or_else(|| {
        format!(
            "kv_head_shard_byte_range: row_bytes ({}) * head_dim ({}) overflowed usize",
            row_bytes, head_dim
        )
    })?;
    let start = shard_bytes.checked_mul(kvh).ok_or_else(|| {
        format!(
            "kv_head_shard_byte_range: shard_bytes ({}) * kvh ({}) overflowed usize",
            shard_bytes, kvh
        )
    })?;
    let end = start.checked_add(shard_bytes).ok_or_else(|| {
        format!(
            "kv_head_shard_byte_range: start ({}) + shard_bytes ({}) overflowed usize",
            start, shard_bytes
        )
    })?;
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- row_bytes_per_row --------

    #[test]
    fn row_bytes_q4k_qwen35_0_8b() {
        // Qwen3.5 0.8B: hidden = 896, Q4_K. 896 / 256 = 3.5 → not multiple. Err.
        let err = row_bytes_per_row(QuantType::Q4K, 896).unwrap_err();
        assert!(
            err.contains("not a multiple"),
            "expected 'not a multiple' error, got: {err}"
        );
    }

    #[test]
    fn row_bytes_q4k_typical_dims() {
        // 512 / 256 = 2 → 2 * 144 = 288
        assert_eq!(row_bytes_per_row(QuantType::Q4K, 512).unwrap(), 288);
        // 1024 / 256 = 4 → 4 * 144 = 576
        assert_eq!(row_bytes_per_row(QuantType::Q4K, 1024).unwrap(), 576);
        // 2048 / 256 = 8 → 8 * 144 = 1152
        assert_eq!(row_bytes_per_row(QuantType::Q4K, 2048).unwrap(), 1152);
    }

    #[test]
    fn row_bytes_q5k_typical_dims() {
        // 1024 / 256 = 4 → 4 * 176 = 704
        assert_eq!(row_bytes_per_row(QuantType::Q5K, 1024).unwrap(), 704);
    }

    #[test]
    fn row_bytes_q6k_typical_dims() {
        // 1024 / 256 = 4 → 4 * 210 = 840
        assert_eq!(row_bytes_per_row(QuantType::Q6K, 1024).unwrap(), 840);
    }

    #[test]
    fn row_bytes_q8_0_typical_dims() {
        // 1024 / 32 = 32 → 32 * 34 = 1088
        assert_eq!(row_bytes_per_row(QuantType::Q8_0, 1024).unwrap(), 1088);
        // Q8_0 only requires multiple of 32, not 256
        assert_eq!(row_bytes_per_row(QuantType::Q8_0, 256).unwrap(), 8 * 34);
    }

    #[test]
    fn row_bytes_zero_hidden_errors_for_all_quants() {
        for q in [
            QuantType::Q4K,
            QuantType::Q5K,
            QuantType::Q6K,
            QuantType::Q8_0,
        ] {
            let err = row_bytes_per_row(q, 0).unwrap_err();
            assert!(
                err.contains("hidden must be > 0"),
                "quant={:?} err={}",
                q,
                err
            );
        }
    }

    #[test]
    fn row_bytes_unaligned_hidden_errors() {
        // Q4_K rejects non-multiple-of-256
        assert!(row_bytes_per_row(QuantType::Q4K, 100)
            .unwrap_err()
            .contains("not a multiple"));
        // Q8_0 rejects non-multiple-of-32
        assert!(row_bytes_per_row(QuantType::Q8_0, 33)
            .unwrap_err()
            .contains("not a multiple"));
    }

    // -------- kv_head_shard_byte_range — per-quant correctness --------

    #[test]
    fn shard_range_q4k_two_kv_heads() {
        // num_kv_heads=2, head_dim=128, hidden=1024, Q4_K
        // row_bytes = 1024/256 * 144 = 4 * 144 = 576
        // shard_bytes = row_bytes * head_dim = 576 * 128 = 73728
        let row_bytes = 576usize;
        let shard_bytes = row_bytes * 128;
        let r0 = kv_head_shard_byte_range(2, 128, 1024, QuantType::Q4K, 0).unwrap();
        assert_eq!(r0, 0..shard_bytes);
        let r1 = kv_head_shard_byte_range(2, 128, 1024, QuantType::Q4K, 1).unwrap();
        assert_eq!(r1, shard_bytes..(2 * shard_bytes));
    }

    #[test]
    fn shard_range_q5k_two_kv_heads() {
        // hidden=1024 → row_bytes = 4 * 176 = 704; head_dim=128 → shard=90112
        let shard_bytes = 704usize * 128;
        let r0 = kv_head_shard_byte_range(2, 128, 1024, QuantType::Q5K, 0).unwrap();
        let r1 = kv_head_shard_byte_range(2, 128, 1024, QuantType::Q5K, 1).unwrap();
        assert_eq!(r0, 0..shard_bytes);
        assert_eq!(r1, shard_bytes..(2 * shard_bytes));
    }

    #[test]
    fn shard_range_q6k_four_kv_heads() {
        // hidden=512 → row_bytes = 2 * 210 = 420; head_dim=64 → shard=26880
        let shard_bytes = 420usize * 64;
        for kvh in 0..4 {
            let r = kv_head_shard_byte_range(4, 64, 512, QuantType::Q6K, kvh).unwrap();
            assert_eq!(r, (kvh * shard_bytes)..((kvh + 1) * shard_bytes));
        }
    }

    #[test]
    fn shard_range_q8_0_eight_kv_heads() {
        // hidden=512 → row_bytes = 16 * 34 = 544; head_dim=128 → shard=69632
        let shard_bytes = 544usize * 128;
        for kvh in 0..8 {
            let r = kv_head_shard_byte_range(8, 128, 512, QuantType::Q8_0, kvh).unwrap();
            assert_eq!(r, (kvh * shard_bytes)..((kvh + 1) * shard_bytes));
        }
    }

    // -------- boundary errors --------

    #[test]
    fn shard_range_kvh_equal_to_num_kv_heads_errors() {
        let err = kv_head_shard_byte_range(2, 128, 1024, QuantType::Q4K, 2).unwrap_err();
        assert!(
            err.contains("out of range"),
            "expected 'out of range' error, got: {err}"
        );
    }

    #[test]
    fn shard_range_kvh_above_num_kv_heads_errors() {
        let err = kv_head_shard_byte_range(2, 128, 1024, QuantType::Q4K, 99).unwrap_err();
        assert!(
            err.contains("out of range"),
            "expected 'out of range' error, got: {err}"
        );
    }

    // -------- shape errors --------

    #[test]
    fn shard_range_zero_num_kv_heads_errors() {
        let err = kv_head_shard_byte_range(0, 128, 1024, QuantType::Q4K, 0).unwrap_err();
        assert!(err.contains("num_kv_heads must be > 0"), "got: {err}");
    }

    #[test]
    fn shard_range_zero_head_dim_errors() {
        let err = kv_head_shard_byte_range(2, 0, 1024, QuantType::Q4K, 0).unwrap_err();
        assert!(err.contains("head_dim must be > 0"), "got: {err}");
    }

    #[test]
    fn shard_range_zero_hidden_errors() {
        let err = kv_head_shard_byte_range(2, 128, 0, QuantType::Q4K, 0).unwrap_err();
        assert!(err.contains("hidden must be > 0"), "got: {err}");
    }

    #[test]
    fn shard_range_unaligned_hidden_errors_q4k() {
        // hidden=100 not multiple of 256 → row_bytes_per_row delegated err
        let err = kv_head_shard_byte_range(2, 128, 100, QuantType::Q4K, 0).unwrap_err();
        assert!(
            err.contains("not a multiple"),
            "expected 'not a multiple' error, got: {err}"
        );
    }

    // -------- mathematical invariants --------

    #[test]
    fn shard_ranges_are_contiguous_no_gap_no_overlap() {
        // Q4_K 4 heads, head_dim=128, hidden=2048
        let num_kv_heads = 4usize;
        let head_dim = 128usize;
        let hidden = 2048usize;
        let quant = QuantType::Q4K;

        let mut prev_end = 0usize;
        for kvh in 0..num_kv_heads {
            let r = kv_head_shard_byte_range(num_kv_heads, head_dim, hidden, quant, kvh).unwrap();
            assert_eq!(
                r.start, prev_end,
                "kvh={kvh} expected start={prev_end}, got {}",
                r.start
            );
            assert!(r.end > r.start, "kvh={kvh} empty/inverted range {r:?}");
            prev_end = r.end;
        }

        // Total = num_kv_heads * head_dim * row_bytes
        let row_bytes = row_bytes_per_row(quant, hidden).unwrap();
        let expected_total = num_kv_heads * head_dim * row_bytes;
        assert_eq!(prev_end, expected_total);
    }

    #[test]
    fn shard_ranges_total_matches_combined_tensor_bytes() {
        // Sum of all shards == size of combined `[num_kv_heads*head_dim, hidden]` tensor.
        // Test for each quant.
        let cases = [
            (QuantType::Q4K, 2usize, 128usize, 1024usize),
            (QuantType::Q5K, 4, 64, 512),
            (QuantType::Q6K, 8, 128, 1024),
            (QuantType::Q8_0, 2, 128, 512),
        ];
        for (quant, num_kv_heads, head_dim, hidden) in cases {
            let row_bytes = row_bytes_per_row(quant, hidden).unwrap();
            let combined_total = num_kv_heads * head_dim * row_bytes;

            let mut sum = 0usize;
            for kvh in 0..num_kv_heads {
                let r =
                    kv_head_shard_byte_range(num_kv_heads, head_dim, hidden, quant, kvh).unwrap();
                sum += r.end - r.start;
            }
            assert_eq!(
                sum, combined_total,
                "quant={quant:?} sum of shard sizes != combined tensor bytes"
            );
        }
    }

    #[test]
    fn single_kv_head_covers_full_tensor() {
        // num_kv_heads=1 → kvh=0 covers entire tensor [0..total).
        let r = kv_head_shard_byte_range(1, 128, 1024, QuantType::Q4K, 0).unwrap();
        let row_bytes = row_bytes_per_row(QuantType::Q4K, 1024).unwrap();
        assert_eq!(r, 0..(128 * row_bytes));
    }

    // -------- cross-validation against rnb-cpu (dev-dep) --------

    #[test]
    fn block_byte_constants_match_rnb_cpu() {
        // QuantType is the canonical block geometry for Vulkan. Keep it aligned
        // with rnb-cpu's concrete block layouts.
        use core::mem::size_of;
        use rnb_cpu::quantize::blocks::{
            BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_0, QK8_0 as RNB_QK8_0, QK_K as RNB_QK_K,
        };

        assert_eq!(size_of::<BlockQ4_K>(), QuantType::Q4K.block_bytes());
        assert_eq!(size_of::<BlockQ5_K>(), QuantType::Q5K.block_bytes());
        assert_eq!(size_of::<BlockQ6_K>(), QuantType::Q6K.block_bytes());
        assert_eq!(size_of::<BlockQ8_0>(), QuantType::Q8_0.block_bytes());
        assert_eq!(RNB_QK_K, QuantType::Q4K.block_elements());
        assert_eq!(RNB_QK8_0, QuantType::Q8_0.block_elements());
    }
}
