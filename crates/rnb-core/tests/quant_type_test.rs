use rnb_core::tensor::QuantType;

#[test]
fn test_new_raw_variants_exist() {
    assert_eq!(QuantType::RawQ5K as u8, 7);
    assert_eq!(QuantType::RawQ6K as u8, 8);
    assert_eq!(QuantType::RawQ8_0 as u8, 9);
    assert_eq!(QuantType::RawQ2KTileGU as u8, 10);
    assert_eq!(QuantType::RawBF16 as u8, 11);
    assert_eq!(QuantType::Q80Pair as u8, 12);
    assert_eq!(QuantType::Q4KCompact as u8, 14);
}

#[test]
fn test_quant_type_from_u8() {
    assert_eq!(QuantType::from_raw_u8(7), Some(QuantType::RawQ5K));
    assert_eq!(QuantType::from_raw_u8(8), Some(QuantType::RawQ6K));
    assert_eq!(QuantType::from_raw_u8(9), Some(QuantType::RawQ8_0));
    assert_eq!(QuantType::from_raw_u8(10), Some(QuantType::RawQ2KTileGU));
    assert_eq!(QuantType::from_raw_u8(11), Some(QuantType::RawBF16));
    assert_eq!(QuantType::from_raw_u8(12), Some(QuantType::Q80Pair));
    assert_eq!(QuantType::from_raw_u8(14), Some(QuantType::Q4KCompact));
    // Q4KRawMeta (=15) was removed when standalone .rnb format was deprecated.
    assert_eq!(QuantType::from_raw_u8(15), None);
}

#[test]
fn q80_tile8_no_longer_exists() {
    // Q80Tile8 (=13) was removed when standalone .rnb was deprecated
    // (in3 phase 1, Task 16). The synthetic tied-output mmap encoder it
    // backed is no longer part of the converter.
    assert!(QuantType::from_raw_u8(13).is_none());
}
