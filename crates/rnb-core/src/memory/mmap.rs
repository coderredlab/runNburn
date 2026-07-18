use std::path::Path;

use memmap2::Mmap;

use crate::error::Result;
use crate::tensor::FileMmapStorage;

pub struct MmapLoader;

impl MmapLoader {
    pub fn load(path: &Path) -> Result<Mmap> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(mmap)
    }

    pub fn load_file_backed(path: &Path) -> Result<FileMmapStorage> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(FileMmapStorage::new(mmap, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_load() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        file.flush().unwrap();
        let mmap = MmapLoader::load(file.path()).unwrap();
        assert_eq!(mmap.len(), 8);
        assert_eq!(&mmap[0..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_mmap_nonexistent() {
        let result = MmapLoader::load(std::path::Path::new("/nonexistent/file"));
        assert!(result.is_err());
    }
}
