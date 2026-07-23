use crate::error::LoaderError;
use crate::gguf::types::GGUFValue;

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
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::U32(n) => Ok(*n),
                GGUFValue::I32(n) => Ok(*n as u32),
                GGUFValue::U64(n) => Ok(*n as u32),
                GGUFValue::I64(n) => Ok(*n as u32),
                GGUFValue::U16(n) => Ok(*n as u32),
                GGUFValue::I16(n) => Ok(*n as u32),
                GGUFValue::U8(n) => Ok(*n as u32),
                GGUFValue::I8(n) => Ok(*n as u32),
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "U32".to_string(),
                }),
            };
        }
    }
    Err(LoaderError::MissingKey(key.to_string()))
}

pub fn get_u64(metadata: &[(String, GGUFValue)], key: &str) -> Result<u64, LoaderError> {
    for (k, v) in metadata {
        if k == key {
            return match v {
                GGUFValue::U64(n) => Ok(*n),
                GGUFValue::U32(n) => Ok(*n as u64),
                GGUFValue::I64(n) => Ok(*n as u64),
                GGUFValue::I32(n) => Ok(*n as u64),
                GGUFValue::U16(n) => Ok(*n as u64),
                GGUFValue::I16(n) => Ok(*n as u64),
                GGUFValue::U8(n) => Ok(*n as u64),
                GGUFValue::I8(n) => Ok(*n as u64),
                _ => Err(LoaderError::TypeMismatch {
                    key: key.to_string(),
                    expected: "U64".to_string(),
                }),
            };
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

/// key가 없으면 None 반환 (optional 필드용)
pub fn get_u32_opt(metadata: &[(String, GGUFValue)], key: &str) -> Option<u32> {
    get_u32(metadata, key).ok()
}

pub fn get_f32_opt(metadata: &[(String, GGUFValue)], key: &str) -> Option<f32> {
    get_f32(metadata, key).ok()
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

pub fn get_bool_opt(metadata: &[(String, GGUFValue)], key: &str) -> Option<bool> {
    get_bool(metadata, key).ok()
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
                        match item {
                            GGUFValue::U32(v) => result.push(*v),
                            GGUFValue::I32(v) => result.push(*v as u32),
                            GGUFValue::U64(v) => result.push(*v as u32),
                            GGUFValue::I64(v) => result.push(*v as u32),
                            GGUFValue::U16(v) => result.push(*v as u32),
                            GGUFValue::I16(v) => result.push(*v as u32),
                            GGUFValue::U8(v) => result.push(*v as u32),
                            GGUFValue::I8(v) => result.push(*v as u32),
                            _ => {
                                return Err(LoaderError::TypeMismatch {
                                    key: key.to_string(),
                                    expected: "Array<U32>".to_string(),
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
    fn test_get_u32_opt_missing() {
        let meta: Vec<(String, GGUFValue)> = vec![];
        assert_eq!(get_u32_opt(&meta, "missing"), None);
    }
}
