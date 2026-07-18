pub const SHADOW_Q2K_TILE_ROWS: usize = 8;
pub const SHADOW_Q2K_BLOCK_BYTES: usize = 84;
pub const SHADOW_Q2K_BLOCK_ELEMS: usize = 256;

#[inline]
pub fn q2k_row_bytes(cols: usize) -> usize {
    assert_eq!(
        cols % SHADOW_Q2K_BLOCK_ELEMS,
        0,
        "cols must be multiple of 256"
    );
    (cols / SHADOW_Q2K_BLOCK_ELEMS) * SHADOW_Q2K_BLOCK_BYTES
}

#[inline]
pub fn q2k_gate_up_tile_bytes_per_expert(n_ff: usize, n_embd: usize) -> usize {
    assert_eq!(
        n_ff % SHADOW_Q2K_TILE_ROWS,
        0,
        "n_ff must be multiple of tile rows"
    );
    n_ff * q2k_row_bytes(n_embd) * 2
}

pub fn pack_q2k_gate_up_tile(gate: &[u8], up: &[u8], n_ff: usize, n_embd: usize) -> Vec<u8> {
    let row_bytes = q2k_row_bytes(n_embd);
    let blocks_per_row = n_embd / SHADOW_Q2K_BLOCK_ELEMS;
    assert_eq!(gate.len(), n_ff * row_bytes);
    assert_eq!(up.len(), n_ff * row_bytes);
    assert_eq!(
        n_ff % SHADOW_Q2K_TILE_ROWS,
        0,
        "n_ff must be multiple of tile rows"
    );

    let mut out = Vec::with_capacity(q2k_gate_up_tile_bytes_per_expert(n_ff, n_embd));
    for tile_start in (0..n_ff).step_by(SHADOW_Q2K_TILE_ROWS) {
        for bi in 0..blocks_per_row {
            let block_off = bi * SHADOW_Q2K_BLOCK_BYTES;
            for lr in 0..SHADOW_Q2K_TILE_ROWS {
                let row = tile_start + lr;
                let row_off = row * row_bytes + block_off;
                out.extend_from_slice(&gate[row_off..row_off + SHADOW_Q2K_BLOCK_BYTES]);
                out.extend_from_slice(&up[row_off..row_off + SHADOW_Q2K_BLOCK_BYTES]);
            }
        }
    }
    out
}

pub fn unpack_q2k_gate_up_tile(packed: &[u8], n_ff: usize, n_embd: usize) -> (Vec<u8>, Vec<u8>) {
    let row_bytes = q2k_row_bytes(n_embd);
    let blocks_per_row = n_embd / SHADOW_Q2K_BLOCK_ELEMS;
    let expected = q2k_gate_up_tile_bytes_per_expert(n_ff, n_embd);
    assert_eq!(packed.len(), expected);

    let mut gate = vec![0u8; n_ff * row_bytes];
    let mut up = vec![0u8; n_ff * row_bytes];
    let mut src = 0usize;
    for tile_start in (0..n_ff).step_by(SHADOW_Q2K_TILE_ROWS) {
        for bi in 0..blocks_per_row {
            let block_off = bi * SHADOW_Q2K_BLOCK_BYTES;
            for lr in 0..SHADOW_Q2K_TILE_ROWS {
                let row = tile_start + lr;
                let row_off = row * row_bytes + block_off;
                gate[row_off..row_off + SHADOW_Q2K_BLOCK_BYTES]
                    .copy_from_slice(&packed[src..src + SHADOW_Q2K_BLOCK_BYTES]);
                src += SHADOW_Q2K_BLOCK_BYTES;
                up[row_off..row_off + SHADOW_Q2K_BLOCK_BYTES]
                    .copy_from_slice(&packed[src..src + SHADOW_Q2K_BLOCK_BYTES]);
                src += SHADOW_Q2K_BLOCK_BYTES;
            }
        }
    }
    (gate, up)
}

#[cfg(test)]
mod tests {
    use super::{pack_q2k_gate_up_tile, unpack_q2k_gate_up_tile, SHADOW_Q2K_TILE_ROWS};

    #[test]
    fn q2k_gate_up_tile_roundtrip_preserves_row_major_order() {
        let n_ff = SHADOW_Q2K_TILE_ROWS * 2;
        let n_embd = 512usize;
        let row_bytes = (n_embd / 256) * 84;
        let total_bytes = n_ff * row_bytes;

        let mut gate = vec![0u8; total_bytes];
        let mut up = vec![0u8; total_bytes];
        for (i, b) in gate.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        for (i, b) in up.iter_mut().enumerate() {
            *b = (255 - (i % 251)) as u8;
        }

        let packed = pack_q2k_gate_up_tile(&gate, &up, n_ff, n_embd);
        let (gate_rt, up_rt) = unpack_q2k_gate_up_tile(&packed, n_ff, n_embd);

        assert_eq!(gate_rt, gate);
        assert_eq!(up_rt, up);
    }
}
