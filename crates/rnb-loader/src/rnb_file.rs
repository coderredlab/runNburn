// .rnb 파일 포맷 (헤더, tensor table, R/W)

use std::io::{self, Seek, SeekFrom, Write};

/// Dense/standalone `.rnb` container magic.
pub const MAGIC: [u8; 4] = *b"RNBD";
pub const VERSION: u32 = 2;

/// 파일 헤더 (64 bytes)
///
/// 레이아웃 (repr(C)):
///   magic[4] + version[4] + num_tensors[4] + _pad0[4]  = 16
///   tensor_table_offset[8]                              = 24
///   data_offset[8]                                      = 32
///   metadata_offset[8]                                  = 40
///   metadata_len[8]                                     = 48
///   _padding[16]                                        = 64
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RnbHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub num_tensors: u32,
    pub _pad0: u32,
    // u64 align이므로 여기서 4바이트 묵시적 패딩이 생기지 않도록
    // _pad0을 앞에 두고 u64 4개가 이어짐
    pub tensor_table_offset: u64,
    pub data_offset: u64,
    pub metadata_offset: u64,
    pub metadata_len: u64,
    pub _padding: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<RnbHeader>() == 64);

/// Tensor entry (96 bytes per entry)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorEntry {
    pub name: [u8; 64],
    pub quant_type: u8,
    pub _pad: [u8; 3],
    pub rows: u32,
    pub cols: u32,
    pub tile_nr: u8,
    pub _pad2: [u8; 3],
    pub data_offset: u64,
    pub data_len: u64,
}

const _: () = assert!(std::mem::size_of::<TensorEntry>() == 96);

/// .rnb 파일 쓰기
///
/// tensors: (name, quant_type, rows, cols, packed_data)
pub fn write_rnb(
    path: &std::path::Path,
    tensors: &[(String, rnb_core::tensor::QuantType, usize, usize, Vec<u8>)],
    metadata: &[u8],
) -> io::Result<()> {
    use std::fs::File;

    let mut f = File::create(path)?;

    let num_tensors = tensors.len() as u32;

    // ── 1. 헤더 placeholder (나중에 seek back해서 채움) ──────────────────
    let header_placeholder = [0u8; 64];
    f.write_all(&header_placeholder)?;

    // ── 2. Tensor table ───────────────────────────────────────────────────
    let tensor_table_offset = 64u64; // 헤더 바로 뒤

    // 각 텐서 데이터 오프셋을 먼저 계산해야 함 → data section 시작 위치를 알아야 함
    // tensor table size = num_tensors * 96
    let tensor_table_size = num_tensors as u64 * 96;
    let pre_align_pos = 64 + tensor_table_size;

    // 4096-byte align
    let data_offset = align_up(pre_align_pos, 4096);

    // 각 텐서 data_offset 계산
    let mut tensor_data_offsets = Vec::with_capacity(tensors.len());
    let mut cur = 0u64;
    for (_, _, _, _, data) in tensors {
        tensor_data_offsets.push(cur);
        cur += data.len() as u64;
    }
    let total_data_len = cur;

    // tensor table 쓰기
    for (i, (name, quant_type, rows, cols, data)) in tensors.iter().enumerate() {
        let mut name_buf = [0u8; 64];
        let bytes = name.as_bytes();
        let copy_len = bytes.len().min(63);
        name_buf[..copy_len].copy_from_slice(&bytes[..copy_len]);

        let entry = TensorEntry {
            name: name_buf,
            quant_type: *quant_type as u8,
            _pad: [0; 3],
            rows: *rows as u32,
            cols: *cols as u32,
            tile_nr: 8,
            _pad2: [0; 3],
            data_offset: tensor_data_offsets[i],
            data_len: data.len() as u64,
        };

        // SAFETY: TensorEntry는 repr(C), POD
        let entry_bytes = unsafe {
            std::slice::from_raw_parts(
                &entry as *const TensorEntry as *const u8,
                std::mem::size_of::<TensorEntry>(),
            )
        };
        f.write_all(entry_bytes)?;
    }

    // ── 3. 4096-byte 정렬 패딩 ───────────────────────────────────────────
    let cur_pos = f.stream_position()?;
    assert_eq!(cur_pos, pre_align_pos);
    let pad_len = (data_offset - pre_align_pos) as usize;
    if pad_len > 0 {
        let pad = vec![0u8; pad_len];
        f.write_all(&pad)?;
    }

    // ── 4. Packed weight data 쓰기 ────────────────────────────────────────
    assert_eq!(f.stream_position()?, data_offset);
    for (_, _, _, _, data) in tensors {
        f.write_all(data)?;
    }

    // ── 5. Metadata 쓰기 ─────────────────────────────────────────────────
    let metadata_offset = data_offset + total_data_len;
    assert_eq!(f.stream_position()?, metadata_offset);
    f.write_all(metadata)?;

    // ── 6. 헤더 seek back해서 채우기 ─────────────────────────────────────
    let header = RnbHeader {
        magic: MAGIC,
        version: VERSION,
        num_tensors,
        _pad0: 0,
        tensor_table_offset,
        data_offset,
        metadata_offset,
        metadata_len: metadata.len() as u64,
        _padding: [0; 16],
    };

    f.seek(SeekFrom::Start(0))?;
    // SAFETY: RnbHeader는 repr(C), POD
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const RnbHeader as *const u8,
            std::mem::size_of::<RnbHeader>(),
        )
    };
    f.write_all(header_bytes)?;

    f.flush()?;
    Ok(())
}

fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_core::tensor::QuantType;

    #[test]
    fn header_size() {
        assert_eq!(std::mem::size_of::<RnbHeader>(), 64);
    }

    #[test]
    fn tensor_entry_size() {
        assert_eq!(std::mem::size_of::<TensorEntry>(), 96);
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rnb_fmt_test_{tag}_{}.rnb", std::process::id()))
    }

    #[test]
    fn write_empty() {
        let path = temp_path("empty");
        write_rnb(&path, &[], &[]).unwrap();
        let data = std::fs::read(&path).unwrap();
        // 헤더 64바이트는 있어야 함
        assert!(data.len() >= 64);
        // 매직 확인
        assert_eq!(&data[..4], b"RNBD");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_single_tensor() {
        let path = temp_path("single");
        let packed_data = vec![0xAAu8; 256];
        let tensors = vec![(
            "test.weight".to_string(),
            QuantType::Q4K,
            8usize,
            4usize,
            packed_data,
        )];
        write_rnb(&path, &tensors, b"hello meta").unwrap();
        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[..4], b"RNBD");
        let _ = std::fs::remove_file(&path);
    }
}
