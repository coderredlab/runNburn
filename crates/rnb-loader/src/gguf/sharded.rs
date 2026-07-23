use crate::convert::map_tensors;
use crate::error::LoaderError;
use crate::gguf::parser::GGUFFile;
use crate::gguf::types::{GGMLType, GGUFValue};
use rnb_core::memory::MmapLoader;
use rnb_core::tensor::{FileMmapStorage, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SPLIT_NO_KEY: &str = "split.no";
const SPLIT_COUNT_KEY: &str = "split.count";
const SPLIT_TENSORS_COUNT_KEY: &str = "split.tensors.count";

pub(crate) struct MappedGguf {
    pub metadata: Vec<(String, GGUFValue)>,
    pub weights: HashMap<String, Tensor>,
    pub float_shapes: HashMap<String, Vec<usize>>,
    pub tensor_ggml_types: HashMap<String, GGMLType>,
    pub tensor_file_offsets: HashMap<String, usize>,
}

pub(crate) fn load_mapped_gguf(path: &Path) -> Result<MappedGguf, LoaderError> {
    let (selected_mmap, selected_gguf) = read_shard(path)?;
    let split_count = split_count(&selected_gguf.metadata)?;

    if split_count == 1 {
        return map_single_shard(selected_gguf, selected_mmap);
    }

    let selected_no = required_split_value(&selected_gguf.metadata, SPLIT_NO_KEY)?;
    if selected_no >= split_count {
        return Err(split_error(format!(
            "{SPLIT_NO_KEY} {selected_no} is outside {SPLIT_COUNT_KEY} {split_count}"
        )));
    }
    let expected_tensors = required_split_value(&selected_gguf.metadata, SPLIT_TENSORS_COUNT_KEY)?;
    let shard_paths = resolve_shard_paths(path, selected_no, split_count)?;

    let mut selected = Some((selected_mmap, selected_gguf));
    let mut metadata = None;
    let mut weights = HashMap::new();
    let mut float_shapes = HashMap::new();
    let mut tensor_ggml_types = HashMap::new();
    let mut tensor_file_offsets = HashMap::new();

    for (shard_no, shard_path) in shard_paths.iter().enumerate() {
        let (mmap, gguf) = if shard_no == selected_no {
            selected.take().expect("selected GGUF shard consumed once")
        } else {
            read_shard(shard_path).map_err(|error| {
                split_error(format!(
                    "failed to load shard '{}': {error}",
                    shard_path.display()
                ))
            })?
        };
        validate_shard(&gguf, shard_no, split_count, expected_tensors)?;

        let (shard_weights, shard_shapes, shard_types, shard_offsets) = map_tensors(&gguf, mmap)?;
        for name in shard_weights.keys() {
            if weights.contains_key(name) {
                return Err(split_error(format!(
                    "tensor '{name}' appears in more than one shard"
                )));
            }
        }

        if shard_no == 0 {
            metadata = Some(gguf.metadata);
        }
        weights.extend(shard_weights);
        float_shapes.extend(shard_shapes);
        tensor_ggml_types.extend(shard_types);
        tensor_file_offsets.extend(shard_offsets);
    }

    if weights.len() != expected_tensors {
        return Err(split_error(format!(
            "{SPLIT_TENSORS_COUNT_KEY} declares {expected_tensors} tensors but {} were loaded",
            weights.len()
        )));
    }

    Ok(MappedGguf {
        metadata: metadata.expect("split GGUF always includes shard zero"),
        weights,
        float_shapes,
        tensor_ggml_types,
        tensor_file_offsets,
    })
}

fn map_single_shard(gguf: GGUFFile, mmap: FileMmapStorage) -> Result<MappedGguf, LoaderError> {
    let (weights, float_shapes, tensor_ggml_types, tensor_file_offsets) = map_tensors(&gguf, mmap)?;
    Ok(MappedGguf {
        metadata: gguf.metadata,
        weights,
        float_shapes,
        tensor_ggml_types,
        tensor_file_offsets,
    })
}

fn read_shard(path: &Path) -> Result<(FileMmapStorage, GGUFFile), LoaderError> {
    let mmap = MmapLoader::load_file_backed(path)?;
    let gguf = GGUFFile::parse(mmap.as_slice())?;
    Ok((mmap, gguf))
}

fn validate_shard(
    gguf: &GGUFFile,
    expected_no: usize,
    expected_count: usize,
    expected_tensors: usize,
) -> Result<(), LoaderError> {
    let shard_no = required_split_value(&gguf.metadata, SPLIT_NO_KEY)?;
    let shard_count = required_split_value(&gguf.metadata, SPLIT_COUNT_KEY)?;
    let shard_tensors = required_split_value(&gguf.metadata, SPLIT_TENSORS_COUNT_KEY)?;

    if shard_no != expected_no {
        return Err(split_error(format!(
            "expected shard {expected_no}, but its metadata declares {SPLIT_NO_KEY} {shard_no}"
        )));
    }
    if shard_count != expected_count {
        return Err(split_error(format!(
            "shard {shard_no} declares {SPLIT_COUNT_KEY} {shard_count}, expected {expected_count}"
        )));
    }
    if shard_tensors != expected_tensors {
        return Err(split_error(format!(
            "shard {shard_no} declares {SPLIT_TENSORS_COUNT_KEY} {shard_tensors}, expected {expected_tensors}"
        )));
    }
    Ok(())
}

fn split_count(metadata: &[(String, GGUFValue)]) -> Result<usize, LoaderError> {
    match split_value(metadata, SPLIT_COUNT_KEY)? {
        Some(0) => Err(split_error(format!("{SPLIT_COUNT_KEY} must be positive"))),
        Some(count) => Ok(count),
        None => Ok(1),
    }
}

fn required_split_value(metadata: &[(String, GGUFValue)], key: &str) -> Result<usize, LoaderError> {
    split_value(metadata, key)?.ok_or_else(|| split_error(format!("missing metadata key {key}")))
}

fn split_value(metadata: &[(String, GGUFValue)], key: &str) -> Result<Option<usize>, LoaderError> {
    let Some((_, value)) = metadata.iter().find(|(name, _)| name == key) else {
        return Ok(None);
    };
    let value = match value {
        GGUFValue::U8(value) => *value as u64,
        GGUFValue::U16(value) => *value as u64,
        GGUFValue::U32(value) => *value as u64,
        GGUFValue::U64(value) => *value,
        GGUFValue::I8(value) if *value >= 0 => *value as u64,
        GGUFValue::I16(value) if *value >= 0 => *value as u64,
        GGUFValue::I32(value) if *value >= 0 => *value as u64,
        GGUFValue::I64(value) if *value >= 0 => *value as u64,
        _ => {
            return Err(split_error(format!(
                "metadata key {key} must be a non-negative integer"
            )))
        }
    };
    usize::try_from(value)
        .map(Some)
        .map_err(|_| split_error(format!("metadata key {key} does not fit usize")))
}

fn resolve_shard_paths(
    selected_path: &Path,
    selected_no: usize,
    split_count: usize,
) -> Result<Vec<PathBuf>, LoaderError> {
    let file_name = selected_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| split_error("split GGUF path must have a UTF-8 file name"))?;
    let stem = file_name
        .strip_suffix(".gguf")
        .ok_or_else(|| split_error("split GGUF file name must end in .gguf"))?;
    let (indexed_stem, file_count) = stem
        .rsplit_once("-of-")
        .ok_or_else(|| split_error("split GGUF file name must end in -00001-of-00001.gguf"))?;
    let (prefix, file_index) = indexed_stem
        .rsplit_once('-')
        .ok_or_else(|| split_error("split GGUF file name is missing its shard index"))?;
    if prefix.is_empty()
        || file_index.is_empty()
        || file_count.is_empty()
        || !file_index.bytes().all(|byte| byte.is_ascii_digit())
        || !file_count.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(split_error(
            "split GGUF file name has an invalid shard suffix",
        ));
    }

    let parsed_index = file_index
        .parse::<usize>()
        .map_err(|_| split_error("split GGUF shard index is too large"))?;
    let parsed_count = file_count
        .parse::<usize>()
        .map_err(|_| split_error("split GGUF shard count is too large"))?;
    if parsed_index != selected_no + 1 || parsed_count != split_count {
        return Err(split_error(format!(
            "file name declares shard {parsed_index} of {parsed_count}, metadata declares shard {} of {split_count}",
            selected_no + 1
        )));
    }

    let parent = selected_path.parent().unwrap_or_else(|| Path::new(""));
    let index_width = file_index.len();
    let count_width = file_count.len();
    Ok((1..=split_count)
        .map(|index| {
            parent.join(format!(
                "{prefix}-{index:0index_width$}-of-{split_count:0count_width$}.gguf"
            ))
        })
        .collect())
}

fn split_error(msg: impl Into<String>) -> LoaderError {
    LoaderError::ParseError {
        offset: 0,
        msg: format!("invalid split GGUF: {}", msg.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parser::tests::{make_gguf_with_tensor, GGUFBuilder};
    use std::fs;

    fn make_split_shard(
        shard_no: u16,
        shard_count: u16,
        total_tensors: u64,
        tensor_name: &str,
        value: f32,
    ) -> Vec<u8> {
        let mut builder = GGUFBuilder::with_counts(3, 1, 4);
        builder.write_string("general.architecture");
        builder.write_u32(8);
        builder.write_string("llama");
        builder.write_string(SPLIT_NO_KEY);
        builder.write_u32(2);
        builder.write_bytes(&shard_no.to_le_bytes());
        builder.write_string(SPLIT_COUNT_KEY);
        builder.write_u32(2);
        builder.write_bytes(&shard_count.to_le_bytes());
        builder.write_string(SPLIT_TENSORS_COUNT_KEY);
        builder.write_u32(12);
        builder.write_u64(total_tensors);

        builder.write_string(tensor_name);
        builder.write_u32(1);
        builder.write_u64(1);
        builder.write_u32(0);
        builder.write_u64(0);
        let padding = (32 - builder.buf.len() % 32) % 32;
        builder.write_bytes(&vec![0; padding]);
        builder.write_bytes(&value.to_le_bytes());
        builder.build()
    }

    #[test]
    fn single_gguf_keeps_existing_direct_mapping_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.gguf");
        fs::write(&path, make_gguf_with_tensor("llama")).expect("write GGUF");

        let mapped = load_mapped_gguf(&path).expect("map direct GGUF");

        assert_eq!(mapped.weights.len(), 1);
        assert!(mapped.weights.contains_key("token_embd.weight"));
    }

    #[test]
    fn split_gguf_maps_every_shard_from_first_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = dir.path().join("model-00001-of-00002.gguf");
        let second = dir.path().join("model-00002-of-00002.gguf");
        fs::write(&first, make_split_shard(0, 2, 2, "first.weight", 1.25))
            .expect("write first shard");
        fs::write(&second, make_split_shard(1, 2, 2, "second.weight", -2.5))
            .expect("write second shard");

        let mapped = load_mapped_gguf(&first).expect("map split GGUF");

        assert_eq!(mapped.weights.len(), 2);
        assert_eq!(
            mapped.weights["first.weight"].as_bytes(),
            Some(1.25f32.to_le_bytes().as_slice())
        );
        assert_eq!(
            mapped.weights["second.weight"].as_bytes(),
            Some((-2.5f32).to_le_bytes().as_slice())
        );
    }

    #[test]
    fn split_gguf_rejects_missing_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = dir.path().join("model-00001-of-00002.gguf");
        fs::write(&first, make_split_shard(0, 2, 2, "first.weight", 1.25))
            .expect("write first shard");

        let error = match load_mapped_gguf(&first) {
            Ok(_) => panic!("missing shard must fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("model-00002-of-00002.gguf"));
    }
}
