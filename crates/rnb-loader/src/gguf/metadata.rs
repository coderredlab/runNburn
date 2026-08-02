use crate::error::LoaderError;
use crate::gguf::types::GGUFValue;

fn integer_as_u64(value: &GGUFValue) -> Option<u64> {
    match value {
        GGUFValue::U8(value) => Some(u64::from(*value)),
        GGUFValue::I8(value) => u64::try_from(*value).ok(),
        GGUFValue::U16(value) => Some(u64::from(*value)),
        GGUFValue::I16(value) => u64::try_from(*value).ok(),
        GGUFValue::U32(value) => Some(u64::from(*value)),
        GGUFValue::I32(value) => u64::try_from(*value).ok(),
        GGUFValue::U64(value) => Some(*value),
        GGUFValue::I64(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

fn integer_as_u32(value: &GGUFValue) -> Option<u32> {
    integer_as_u64(value).and_then(|value| u32::try_from(value).ok())
}

pub fn get_string<'a>(
    metadata: &'a [(String, GGUFValue)],
    key: &str,
) -> Result<&'a str, LoaderError> {
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::String(s) => Ok(s.as_str()),
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "String".to_string(),
                }),
            };
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

pub fn get_u32(metadata: &[(String, GGUFValue)], key: &str) -> Result<u32, LoaderError> {
    for (k, value) in metadata {
        if k == key {
            return integer_as_u32(value).ok_or_else(|| LoaderError::TypeMismatch {
                key: key.to_string(),
                expected: "non-negative integer fitting U32".to_string(),
            });
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

pub fn get_u64(metadata: &[(String, GGUFValue)], key: &str) -> Result<u64, LoaderError> {
    for (k, value) in metadata {
        if k == key {
            return integer_as_u64(value).ok_or_else(|| LoaderError::TypeMismatch {
                key: key.to_string(),
                expected: "non-negative integer fitting U64".to_string(),
            });
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

pub fn get_f32(metadata: &[(String, GGUFValue)], key: &str) -> Result<f32, LoaderError> {
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::F32(f) => Ok(*f),
                GGUFValue::F64(f) => Ok(*f as f32),
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "F32".to_string(),
                }),
            };
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

fn optional<T>(value: Result<T, LoaderError>) -> Result<Option<T>, LoaderError> {
    match value {
        Ok(value) => Ok(Some(value)),
        Err(LoaderError::MissingKey(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

/// key가 없으면 None을 반환하고, 값이 있으면 타입 오류를 보존한다.
pub fn get_u32_opt(
    metadata: &[(String, GGUFValue)],
    key: &str,
) -> Result<Option<u32>, LoaderError> {
    optional(get_u32(metadata, key))
}

pub fn get_f32_opt(
    metadata: &[(String, GGUFValue)],
    key: &str,
) -> Result<Option<f32>, LoaderError> {
    optional(get_f32(metadata, key))
}

pub fn get_bool(metadata: &[(String, GGUFValue)], key: &str) -> Result<bool, LoaderError> {
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::Bool(b) => Ok(*b),
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "Bool".to_string(),
                }),
            };
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

pub fn get_bool_opt(
    metadata: &[(String, GGUFValue)],
    key: &str,
) -> Result<Option<bool>, LoaderError> {
    optional(get_bool(metadata, key))
}

pub fn get_bool_array(
    metadata: &[(String, GGUFValue)],
    key: &str,
) -> Result<Vec<bool>, LoaderError> {
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::Array(items) => {
                    let mut result = Vec::with_capacity(items.len());
                    for item in items {
                        match item {
                            GGUFValue::Bool(b) => result.push(*b),
                            _ => {
                                return Err(LoaderError::TypeMismatch {
                                    key: key.to_string(),
                                    expected: "Array<Bool>".to_string(),
                                })
                            }
                        }
                    }
                    Ok(result)
                }
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "Array".to_string(),
                }),
            };
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

/// Array(U32 or I32) 값을 Vec<u32>로 반환
pub fn get_u32_array(metadata: &[(String, GGUFValue)], key: &str) -> Result<Vec<u32>, LoaderError> {
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::Array(items) => {
                    let mut result = Vec::with_capacity(items.len());
                    for item in items {
                        let value =
                            integer_as_u32(item).ok_or_else(|| LoaderError::TypeMismatch {
                                key: key.to_string(),
                                expected: "Array<non-negative integer fitting U32>".to_string(),
                            })?;
                        result.push(value);
                    }
                    Ok(result)
                }
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "Array".to_string(),
                }),
            };
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

/// Array(String) 값을 Vec<String>으로 반환
pub fn get_string_array(
    metadata: &[(String, GGUFValue)],
    key: &str,
) -> Result<Vec<String>, LoaderError> {
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::Array(items) => {
                    let mut result = Vec::with_capacity(items.len());
                    for item in items {
                        match item {
                            GGUFValue::String(s) => result.push(s.clone()),
                            _ => {
                                return Err(LoaderError::TypeMismatch {
                                    key: key.to_string(),
                                    expected: "Array<String>".to_string(),
                                })
                            }
                        }
                    }
                    Ok(result)
                }
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "Array".to_string(),
                }),
            };
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

/// Array(F32) 값을 Vec<f32>로 반환
pub fn get_f32_array(metadata: &[(String, GGUFValue)], key: &str) -> Result<Vec<f32>, LoaderError> {
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::Array(items) => {
                    let mut result = Vec::with_capacity(items.len());
                    for item in items {
                        match item {
                            GGUFValue::F32(f) => result.push(*f),
                            GGUFValue::F64(f) => result.push(*f as f32),
                            _ => {
                                return Err(LoaderError::TypeMismatch {
                                    key: key.to_string(),
                                    expected: "Array<F32>".to_string(),
                                })
                            }
                        }
                    }
                    Ok(result)
                }
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "Array".to_string(),
                }),
            };
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::types::GGUFValue;

    fn kv(key: &str, val: GGUFValue) -> (String, GGUFValue) {
        (key.to_string(), val)
    }

    #[test]
    fn test_get_string_ok() {
        let meta = vec![kv(
            "general.architecture",
            GGUFValue::String("llama".to_string()),
        )];
        assert_eq!(get_string(&meta, "general.architecture").unwrap(), "llama");
    }

    #[test]
    fn test_get_string_missing() {
        let meta: Vec<(String, GGUFValue)> = vec![];
        assert!(matches!(
            get_string(&meta, "x"),
            Err(LoaderError::MissingKey(_))
        ));
    }

    #[test]
    fn test_get_string_wrong_type() {
        let meta = vec![kv("k", GGUFValue::U32(5))];
        assert!(matches!(
            get_string(&meta, "k"),
            Err(LoaderError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_get_u32_ok() {
        let meta = vec![kv("llama.block_count", GGUFValue::U32(32))];
        assert_eq!(get_u32(&meta, "llama.block_count").unwrap(), 32);
    }

    #[test]
    fn test_get_u32_from_i32() {
        let meta = vec![kv("k", GGUFValue::I32(16))];
        assert_eq!(get_u32(&meta, "k").unwrap(), 16);
    }

    #[test]
    fn test_get_f32_ok() {
        let meta = vec![kv("llama.rope.freq_base", GGUFValue::F32(10000.0))];
        assert!((get_f32(&meta, "llama.rope.freq_base").unwrap() - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn test_get_u32_opt_preserves_missing_and_type_errors() {
        let meta: Vec<(String, GGUFValue)> = vec![];
        assert_eq!(get_u32_opt(&meta, "missing").unwrap(), None);

        let meta = vec![kv("k", GGUFValue::String("wrong".to_string()))];
        assert!(matches!(
            get_u32_opt(&meta, "k"),
            Err(LoaderError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_unsigned_accessors_reject_negative_and_overflowing_values() {
        for value in [
            GGUFValue::I8(-1),
            GGUFValue::I16(-1),
            GGUFValue::I32(-1),
            GGUFValue::I64(-1),
            GGUFValue::U64(u64::from(u32::MAX) + 1),
        ] {
            let meta = vec![kv("k", value)];
            assert!(matches!(
                get_u32(&meta, "k"),
                Err(LoaderError::TypeMismatch { .. })
            ));
        }

        let meta = vec![kv("k", GGUFValue::I64(-1))];
        assert!(matches!(
            get_u64(&meta, "k"),
            Err(LoaderError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_get_u32_array_rejects_negative_and_overflowing_values() {
        for value in [
            GGUFValue::I32(-1),
            GGUFValue::I64(-1),
            GGUFValue::U64(u64::from(u32::MAX) + 1),
        ] {
            let meta = vec![kv("k", GGUFValue::Array(vec![value]))];
            assert!(matches!(
                get_u32_array(&meta, "k"),
                Err(LoaderError::TypeMismatch { .. })
            ));
        }
    }
}
