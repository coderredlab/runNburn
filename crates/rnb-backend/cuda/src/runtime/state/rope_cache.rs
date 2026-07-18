use super::super::*;

impl CudaState {
    pub(in crate::runtime) fn rope_table_ptrs(
        &mut self,
        head_dim: usize,
        seq_len: usize,
        pos_start: usize,
        rope_theta: f32,
    ) -> Result<ResidentRopeTable, String> {
        if head_dim == 0 || head_dim % 2 != 0 || seq_len == 0 {
            return Err(format!(
                "invalid RoPE table shape: head_dim={head_dim} seq_len={seq_len}"
            ));
        }
        let key = RopeTableKey {
            head_dim,
            seq_len,
            pos_start,
            rope_theta_bits: rope_theta.to_bits(),
        };
        if let Some(entry) = self.resident_rope_tables.get(&key) {
            return Ok(*entry);
        }

        let table_len = seq_len.checked_mul(head_dim / 2).ok_or_else(|| {
            format!("RoPE table length overflow: seq_len={seq_len} head_dim={head_dim}")
        })?;
        let bytes = table_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("RoPE table byte overflow: len={table_len}"))?;
        // cu31: host CPU rope_inplace 와 동일 chain 방식 (theta_scale^i 누적)
        // 으로 inv_freq 계산. powf 와 chain 미세 정밀도 차이가 decode wrapper
        // 깨짐 원인 가설 — host 와 bit-identical 보장.
        let theta_scale = rope_theta.powf(-2.0_f32 / head_dim as f32);
        let inv_freq = {
            let mut v = Vec::with_capacity(head_dim / 2);
            let mut freq = 1.0f32;
            for _ in 0..head_dim / 2 {
                v.push(freq);
                freq *= theta_scale;
            }
            v
        };
        let mut rope_sin = Vec::with_capacity(table_len);
        let mut rope_cos = Vec::with_capacity(table_len);
        for token in 0..seq_len {
            let pos = (pos_start + token) as f32;
            for &freq in &inv_freq {
                let (sin_a, cos_a) = (pos * freq).sin_cos();
                rope_sin.push(sin_a);
                rope_cos.push(cos_a);
            }
        }

        let sin_ptr = unsafe { self.api.mem_alloc(bytes) }?;
        let cos_ptr = unsafe { self.api.mem_alloc(bytes) }?;
        unsafe {
            self.api.memcpy_htod_async(
                sin_ptr,
                rope_sin.as_ptr().cast::<libc::c_void>(),
                bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                cos_ptr,
                rope_cos.as_ptr().cast::<libc::c_void>(),
                bytes,
                self.stream,
            )?;
        }
        let entry = ResidentRopeTable {
            sin_ptr,
            cos_ptr,
            bytes,
        };
        self.resident_rope_tables.insert(key, entry);
        Ok(entry)
    }
}
