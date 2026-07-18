/// Q4_K fixture: 고정 결정적 144바이트 블록 + CPU reference dequant sum.
///
/// rnb-cpu 의 검증된 `dequantize_q4_k` 를 재사용해 reference 를 만든다.
/// MSL 커널 정확도 검증용 — 직접 dequant 재구현 금지.

/// 고정 BlockQ4_K 블록을 144 바이트로 반환.
///
/// 레이아웃 (BlockQ4_K repr(C)):
///   offset  0-1  : d     (f16 little-endian)
///   offset  2-3  : dmin  (f16 little-endian)
///   offset  4-15 : scales[12]  (6-bit packed sub-block scales/mins)
///   offset 16-143: qs[128]     (4-bit packed quants, 256 weights)
///
/// 합이 0 이 아니도록: d=1.0(0x3C00), dmin=0.5(0x3800),
/// scales 는 j<4 구간에서 sc=10, m=5, j>=4 구간도 non-zero 패턴.
/// qs 는 nibble 값이 0 이 아닌 반복 패턴 (0x35).
pub fn q4k_block_fixed() -> Vec<u8> {
    let mut block = vec![0u8; 144];

    // d = 1.0 as f16 = 0x3C00 (little-endian: 0x00, 0x3C)
    block[0] = 0x00;
    block[1] = 0x3C;

    // dmin = 0.5 as f16 = 0x3800 (little-endian: 0x00, 0x38)
    block[2] = 0x00;
    block[3] = 0x38;

    // scales[12] 세팅 — get_scale_min_k4 기준:
    //   j < 4: sc = scales[j] & 63, m = scales[j+4] & 63
    //   j >= 4: sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    //           m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
    //
    // 목표: 8 sub-block 전부 sc≠0, m≠0 → 합이 0 이 아님.
    // j=0..3: scales[j] 에 sc 하위6비트 = 10(0b001010), scales[j+4] 에 m 하위6비트 = 5(0b000101)
    // → scales[0..3] = 10 (0x0A), scales[4..7] = 5 (0x05)
    // j=4..7: sc = scales[j+4] & 0x0F → scales[8..11] 하위4비트 = 8(0x8)
    //         m  = scales[j+4] >> 4   → scales[8..11] 상위4비트 = 4(0x4)
    //         → scales[8..11] = 0x48
    //   그리고 scales[j-4=0..3] >> 6 = 0 (상위2비트 = 0 이므로 sc/m 고비트 = 0)
    //   → j=4..7: sc = 8, m = 4
    for i in 0..4 {
        block[4 + i] = 10; // scales[0..3]: low6 = sc for j=0..3
        block[8 + i] = 5; // scales[4..7]: low6 = m for j=0..3
    }
    for i in 0..4 {
        // scales[8..11]: low4 = sc for j=4..7, high4 = m for j=4..7
        // sc_j4 = 8 (0x8), m_j4 = 4 (0x4) → byte = (4 << 4) | 8 = 0x48
        block[12 + i] = 0x48;
    }

    // qs[128] — nibble 패턴 0x35: low=5, high=3 (양쪽 모두 nonzero)
    for i in 0..128 {
        block[16 + i] = 0x35;
    }

    block
}

/// CPU reference: rnb-cpu 의 `dequantize_q4_k` 로 block bytes 를 dequant 한 뒤 합산.
///
/// `block` 은 `q4k_block_fixed()` 가 반환한 144 바이트 슬라이스.
pub fn q4k_dequant_sum(block: &[u8]) -> f32 {
    use rnb_cpu::quantize::{dequantize_q4_k, BlockQ4_K};

    assert_eq!(block.len(), 144, "Q4_K block must be exactly 144 bytes");

    // SAFETY: block.len() == 144 == size_of::<BlockQ4_K>(), repr(C), alignment은
    // f16(align 2)이 최대 — u8 슬라이스는 최소 align 1이라 직접 전송할 때
    // 가장 안전한 방법은 바이트 복사 후 재해석.
    // 여기서는 stack 배열로 복사해 alignment를 보장한다.
    #[repr(C, align(2))]
    struct Aligned144([u8; 144]);
    let mut aligned = Aligned144([0u8; 144]);
    aligned.0.copy_from_slice(block);

    let block_ref: &BlockQ4_K = unsafe { &*(aligned.0.as_ptr() as *const BlockQ4_K) };

    let mut output = [0.0f32; 256];
    dequantize_q4_k(block_ref, &mut output);
    output.iter().sum()
}

/// CPU reference: `dequantize_q4_k` 로 144 바이트 블록을 256개 f32 로 dequant 반환.
/// `q4k_dequant_sum` 과 달리 input 가중 GEMV reference(블록별 내적)에 쓰인다.
pub fn q4k_dequant(block: &[u8]) -> [f32; 256] {
    use rnb_cpu::quantize::{dequantize_q4_k, BlockQ4_K};

    assert_eq!(block.len(), 144, "Q4_K block must be exactly 144 bytes");

    #[repr(C, align(2))]
    struct Aligned144([u8; 144]);
    let mut aligned = Aligned144([0u8; 144]);
    aligned.0.copy_from_slice(block);

    let block_ref: &BlockQ4_K = unsafe { &*(aligned.0.as_ptr() as *const BlockQ4_K) };

    let mut output = [0.0f32; 256];
    dequantize_q4_k(block_ref, &mut output);
    output
}

/// Q4_K weight[N,K]와 f32 input[M,K]의 CPU reference.
///
/// 양자화 포맷 해석은 `q4k_dequant`에만 맡겨 Metal 커널과 독립된 oracle로 쓴다.
pub fn q4k_gemm_reference(wb: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
    assert_eq!(k % 256, 0, "Q4_K K must be a multiple of 256");
    assert_eq!(wb.len(), n * (k / 256) * 144);
    assert_eq!(input.len(), m * k);
    let nb = k / 256;
    let bpr = nb * 144;
    let mut out = vec![0.0f32; m * n];
    let mut deq_row = vec![0.0f32; k];
    for row in 0..n {
        let rb = &wb[row * bpr..(row + 1) * bpr];
        for sb in 0..nb {
            let deq = q4k_dequant(&rb[sb * 144..(sb + 1) * 144]);
            deq_row[sb * 256..(sb + 1) * 256].copy_from_slice(&deq);
        }
        for tok in 0..m {
            let inp = &input[tok * k..(tok + 1) * k];
            out[tok * n + row] = deq_row.iter().zip(inp.iter()).map(|(&w, &x)| w * x).sum();
        }
    }
    out
}

/// 고정 BlockQ6_K 블록을 210 바이트로 반환.
///
/// 레이아웃 (BlockQ6_K repr(C)):
///   offset   0-127 : ql[128]    (low 4 bits, 256 weights → 4비트씩)
///   offset 128-191 : qh[64]     (high 2 bits)
///   offset 192-207 : scales[16] (i8, signed)
///   offset 208-209 : d          (f16 little-endian)
///
/// 합이 0 이 아니도록: d=1.0(0x3C00), scales 전부 8(양수 i8),
/// ql=0x53(low nibble=3, high nibble=5), qh=0x1B(2비트 그룹별 3/2/1/0)
/// → q 값이 sub-group 마다 다양하게 분포(-32 오프셋 후 nonzero).
pub fn q6k_block_fixed() -> Vec<u8> {
    let mut block = vec![0u8; 210];

    // ql[0..128] = 0x53 (low nibble = 3, high nibble = 5)
    for b in block.iter_mut().take(128) {
        *b = 0x53;
    }
    // qh[128..192] = 0x1B = 0b0001_1011
    //   bits 0-1 = 0b11 = 3, bits 2-3 = 0b10 = 2, bits 4-5 = 0b01 = 1, bits 6-7 = 0b00 = 0
    for b in block.iter_mut().skip(128).take(64) {
        *b = 0x1B;
    }
    // scales[192..208] = 8 (i8, 양수)
    for b in block.iter_mut().skip(192).take(16) {
        *b = 8;
    }
    // d = 1.0 as f16 = 0x3C00 (little-endian: 0x00, 0x3C)
    block[208] = 0x00;
    block[209] = 0x3C;

    block
}

/// CPU reference: rnb-cpu 의 검증된 `dequantize_q6_k` 로 210 바이트 블록을 dequant.
///
/// `block` 은 `q6k_block_fixed()` 가 반환한 210 바이트 슬라이스.
/// MSL 커널 정확도 검증용 ground truth — 직접 dequant 재구현 금지.
pub fn q6k_dequant(block: &[u8]) -> [f32; 256] {
    use rnb_cpu::quantize::{dequantize_q6_k, BlockQ6_K};

    assert_eq!(block.len(), 210, "Q6_K block must be exactly 210 bytes");

    // BlockQ6_K 의 최대 alignment 는 f16(align 2). align(2) wrapper 로 복사해
    // alignment 를 보장한 뒤 재해석.
    #[repr(C, align(2))]
    struct Aligned210([u8; 210]);
    let mut aligned = Aligned210([0u8; 210]);
    aligned.0.copy_from_slice(block);

    let block_ref: &BlockQ6_K = unsafe { &*(aligned.0.as_ptr() as *const BlockQ6_K) };

    let mut output = [0.0f32; 256];
    dequantize_q6_k(block_ref, &mut output);
    output
}

pub fn q6k_block_from_parts(d_val: f32, scales: [i8; 16], ql: [u8; 128], qh: [u8; 64]) -> Vec<u8> {
    let mut block = vec![0u8; 210];
    block[0..128].copy_from_slice(&ql);
    block[128..192].copy_from_slice(&qh);
    for (i, &s) in scales.iter().enumerate() {
        block[192 + i] = s as u8;
    }
    block[208..210].copy_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
    block
}

pub fn q6k_rows_pattern(n: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0, "Q6_K K must be a multiple of 256");
    let nb = k / 256;
    let mut out = Vec::with_capacity(n * nb * 210);
    for row in 0..n {
        for blk in 0..nb {
            let seed = row * nb + blk;
            let d_val = 0.02 + (seed % 7) as f32 * 0.004;
            let mut scales = [0i8; 16];
            for (i, s) in scales.iter_mut().enumerate() {
                *s = (((seed + i) % 13) as i32 - 6) as i8;
            }
            let mut ql = [0u8; 128];
            for (i, q) in ql.iter_mut().enumerate() {
                *q = ((seed * 3 + i * 5) % 256) as u8;
            }
            let mut qh = [0u8; 64];
            for (i, q) in qh.iter_mut().enumerate() {
                *q = ((seed * 7 + i * 11) % 256) as u8;
            }
            out.extend(q6k_block_from_parts(d_val, scales, ql, qh));
        }
    }
    out
}

pub fn q6k_gemm_reference(wb: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
    assert_eq!(k % 256, 0, "Q6_K K must be a multiple of 256");
    assert_eq!(wb.len(), n * (k / 256) * 210);
    assert_eq!(input.len(), m * k);
    let nb = k / 256;
    let bpr = nb * 210;
    let mut out = vec![0.0f32; m * n];
    let mut deq_row = vec![0.0f32; k];
    for row in 0..n {
        let rb = &wb[row * bpr..(row + 1) * bpr];
        for sb in 0..nb {
            let deq = q6k_dequant(&rb[sb * 210..(sb + 1) * 210]);
            deq_row[sb * 256..(sb + 1) * 256].copy_from_slice(&deq);
        }
        for tok in 0..m {
            let inp = &input[tok * k..(tok + 1) * k];
            out[tok * n + row] = deq_row.iter().zip(inp.iter()).map(|(&w, &x)| w * x).sum();
        }
    }
    out
}

#[cfg(target_os = "macos")]
pub struct QwenMoeGdnDecodeChainFixture {
    pub hidden: Vec<f32>,
    conv_state: Vec<f32>,
    delta_state: Vec<f32>,
    attn_norm: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    conv1d: Vec<f32>,
    ssm_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    qkv_raw: Vec<u8>,
    gate_raw: Vec<u8>,
    alpha_raw: Vec<u8>,
    beta_raw: Vec<u8>,
    ssm_out_raw: Vec<u8>,
    router_w: Vec<f32>,
    gate_exps_raw: Vec<u8>,
    up_exps_raw: Vec<u8>,
    down_exps_raw: Vec<u8>,
    shared_input_scale: Vec<f32>,
    shared_gate_raw: Vec<u8>,
    shared_up_raw: Vec<u8>,
    shared_down_raw: Vec<u8>,
}

#[cfg(target_os = "macos")]
impl QwenMoeGdnDecodeChainFixture {
    const HIDDEN_DIM: usize = 256;
    const CONV_CHANNELS: usize = 3;
    const CONV_KERNEL: usize = 4;
    const Z_DIM: usize = 1;
    const NUM_V_HEADS: usize = 1;
    const NUM_K_HEADS: usize = 1;
    const HEAD_K_DIM: usize = 1;
    const HEAD_V_DIM: usize = 1;
    const N_FF: usize = 256;
    const N_EXPERT: usize = 4;
    const N_EXPERT_USED: usize = 2;

    pub fn layer0_spec(&self) -> crate::GdnMoeQwenChainSpecRef<'_> {
        self.spec(0)
    }

    pub fn layer1_spec(&self) -> crate::GdnMoeQwenChainSpecRef<'_> {
        self.spec(1)
    }

    fn spec(&self, layer: usize) -> crate::GdnMoeQwenChainSpecRef<'_> {
        crate::GdnMoeQwenChainSpecRef {
            layer,
            conv_state: &self.conv_state,
            delta_state: &self.delta_state,
            attn_norm_weight: &self.attn_norm,
            dt_bias_weight: &self.dt_bias,
            ssm_a_weight: &self.ssm_a,
            conv1d_weight: &self.conv1d,
            ssm_norm_weight: &self.ssm_norm,
            ffn_norm_weight: &self.ffn_norm,
            qkv_raw: &self.qkv_raw,
            gate_raw: &self.gate_raw,
            alpha_raw: &self.alpha_raw,
            beta_raw: &self.beta_raw,
            ssm_out_raw: &self.ssm_out_raw,
            router_w: &self.router_w,
            gate_exps_raw: &self.gate_exps_raw,
            gate_expert_bytes: Self::N_FF * 144,
            up_exps_raw: &self.up_exps_raw,
            up_expert_bytes: Self::N_FF * 144,
            down_exps_raw: &self.down_exps_raw,
            down_expert_bytes: Self::HIDDEN_DIM * 210,
            shared_input_scale: &self.shared_input_scale,
            shared_gate_raw: &self.shared_gate_raw,
            shared_up_raw: &self.shared_up_raw,
            shared_down_raw: &self.shared_down_raw,
            qkv_q: 0,
            gate_q: 0,
            alpha_q: 4,
            beta_q: 4,
            ssm_out_q: 4,
            down_quant: 2,
            hidden_dim: Self::HIDDEN_DIM,
            conv_channels: Self::CONV_CHANNELS,
            conv_kernel: Self::CONV_KERNEL,
            z_dim: Self::Z_DIM,
            num_v_heads: Self::NUM_V_HEADS,
            num_k_heads: Self::NUM_K_HEADS,
            head_k_dim: Self::HEAD_K_DIM,
            head_v_dim: Self::HEAD_V_DIM,
            n_ff: Self::N_FF,
            n_expert: Self::N_EXPERT,
            n_expert_used: Self::N_EXPERT_USED,
            eps: 1e-6,
        }
    }
}

fn repeat_block(block: &[u8], rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(block.len() * rows);
    for _ in 0..rows {
        out.extend_from_slice(block);
    }
    out
}

fn f32_raw(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(target_os = "macos")]
pub fn qwen_moe_gdn_decode_chain_fixture() -> QwenMoeGdnDecodeChainFixture {
    let q4 = q4k_block_fixed();
    let q6 = q6k_block_fixed();
    let hidden = (0..QwenMoeGdnDecodeChainFixture::HIDDEN_DIM)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.001)
        .collect::<Vec<_>>();
    let conv_state_len = (QwenMoeGdnDecodeChainFixture::CONV_KERNEL - 1)
        * QwenMoeGdnDecodeChainFixture::CONV_CHANNELS;
    let delta_state_len = QwenMoeGdnDecodeChainFixture::NUM_V_HEADS
        * QwenMoeGdnDecodeChainFixture::HEAD_V_DIM
        * QwenMoeGdnDecodeChainFixture::HEAD_K_DIM;
    let conv_state = (0..conv_state_len)
        .map(|i| (i as f32 + 1.0) * 0.002)
        .collect::<Vec<_>>();
    let delta_state = (0..delta_state_len)
        .map(|i| (i as f32 + 1.0) * 0.003)
        .collect::<Vec<_>>();
    let ones_hidden = vec![1.0; QwenMoeGdnDecodeChainFixture::HIDDEN_DIM];
    let ones_head = vec![1.0; QwenMoeGdnDecodeChainFixture::HEAD_V_DIM];
    let f32_row_hidden = vec![0.001; QwenMoeGdnDecodeChainFixture::HIDDEN_DIM];
    let f32_ssm_out = vec![0.001; QwenMoeGdnDecodeChainFixture::HIDDEN_DIM];

    QwenMoeGdnDecodeChainFixture {
        hidden,
        conv_state,
        delta_state,
        attn_norm: ones_hidden.clone(),
        dt_bias: vec![0.0; QwenMoeGdnDecodeChainFixture::CONV_CHANNELS],
        ssm_a: vec![0.0; QwenMoeGdnDecodeChainFixture::NUM_V_HEADS],
        conv1d: vec![
            0.001;
            QwenMoeGdnDecodeChainFixture::CONV_CHANNELS
                * QwenMoeGdnDecodeChainFixture::CONV_KERNEL
        ],
        ssm_norm: ones_head,
        ffn_norm: ones_hidden,
        qkv_raw: repeat_block(&q4, QwenMoeGdnDecodeChainFixture::CONV_CHANNELS),
        gate_raw: repeat_block(&q4, QwenMoeGdnDecodeChainFixture::Z_DIM),
        alpha_raw: f32_raw(&f32_row_hidden),
        beta_raw: f32_raw(&f32_row_hidden),
        ssm_out_raw: f32_raw(&f32_ssm_out),
        router_w: vec![
            0.001;
            QwenMoeGdnDecodeChainFixture::N_EXPERT
                * QwenMoeGdnDecodeChainFixture::HIDDEN_DIM
        ],
        gate_exps_raw: repeat_block(
            &q4,
            QwenMoeGdnDecodeChainFixture::N_EXPERT * QwenMoeGdnDecodeChainFixture::N_FF,
        ),
        up_exps_raw: repeat_block(
            &q4,
            QwenMoeGdnDecodeChainFixture::N_EXPERT * QwenMoeGdnDecodeChainFixture::N_FF,
        ),
        down_exps_raw: repeat_block(
            &q6,
            QwenMoeGdnDecodeChainFixture::N_EXPERT * QwenMoeGdnDecodeChainFixture::HIDDEN_DIM,
        ),
        shared_input_scale: vec![1.0; QwenMoeGdnDecodeChainFixture::HIDDEN_DIM],
        shared_gate_raw: repeat_block(&q4, QwenMoeGdnDecodeChainFixture::N_FF),
        shared_up_raw: repeat_block(&q4, QwenMoeGdnDecodeChainFixture::N_FF),
        shared_down_raw: repeat_block(&q6, QwenMoeGdnDecodeChainFixture::HIDDEN_DIM),
    }
}

/// 고정 BlockQ5_K 블록을 176바이트로 반환.
///
/// Q5_K raw 레이아웃을 직접 채우되 half scale은 유한한 양수, packed scale/high/low
/// quant는 결정적 비영 패턴으로 둔다. CPU/GPU dequant 순서 비교용 fixture다.
pub fn q5k_block_fixed() -> Vec<u8> {
    let mut block = vec![0u8; 176];
    block[0..2].copy_from_slice(&half::f16::from_f32(0.03125).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
    for (index, value) in block[4..16].iter_mut().enumerate() {
        *value = ((index * 19 + 7) % 256) as u8;
    }
    for (index, value) in block[16..48].iter_mut().enumerate() {
        *value = ((index * 11 + 3) % 256) as u8;
    }
    for (index, value) in block[48..176].iter_mut().enumerate() {
        *value = ((index * 5 + 13) % 256) as u8;
    }
    block
}

/// CPU reference: rnb-cpu 의 검증된 `dequantize_q5_k` 로 176 바이트 블록을 256 f32 로 dequant.
///
/// Q5_K super-block layout (176 bytes, BlockQ5_K repr(C)):
///   0-1 d / 2-3 dmin / 4-15 scales[12] / 16-47 qh[32] / 48-175 ql[128].
/// MSL coalesced 커널 정확도 검증용 ground truth — 직접 dequant 재구현 금지.
pub fn q5k_dequant(block: &[u8]) -> [f32; 256] {
    use rnb_cpu::quantize::{dequantize_q5_k, BlockQ5_K};

    assert_eq!(block.len(), 176, "Q5_K block must be exactly 176 bytes");

    #[repr(C, align(2))]
    struct Aligned176([u8; 176]);
    let mut aligned = Aligned176([0u8; 176]);
    aligned.0.copy_from_slice(block);

    let block_ref: &BlockQ5_K = unsafe { &*(aligned.0.as_ptr() as *const BlockQ5_K) };

    let mut output = [0.0f32; 256];
    dequantize_q5_k(block_ref, &mut output);
    output
}

/// CPU reference: rnb-cpu 의 검증된 `dequantize_q8_0` 로 34 바이트 블록을 32 f32 로 dequant.
///
/// Q8_0 block layout (34 bytes, BlockQ8_0 repr(C)): 0-1 d(half) / 2-33 qs[32](i8).
/// MSL coalesced 커널 정확도 검증용 ground truth — 직접 dequant 재구현 금지.
pub fn q8_0_dequant(block: &[u8]) -> [f32; 32] {
    use rnb_cpu::quantize::{dequantize_q8_0, BlockQ8_0};

    assert_eq!(block.len(), 34, "Q8_0 block must be exactly 34 bytes");

    #[repr(C, align(2))]
    struct Aligned34([u8; 34]);
    let mut aligned = Aligned34([0u8; 34]);
    aligned.0.copy_from_slice(block);

    let block_ref: &BlockQ8_0 = unsafe { &*(aligned.0.as_ptr() as *const BlockQ8_0) };

    let mut output = [0.0f32; 32];
    dequantize_q8_0(block_ref, &mut output);
    output
}

#[cfg(target_os = "macos")]
pub struct QwenMoeLlamaPrefillLayerChainFixture {
    pub hidden: Vec<f32>,
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_q_raw: Vec<u8>,
    attn_k_raw: Vec<u8>,
    attn_v_raw: Vec<u8>,
    attn_o_raw: Vec<u8>,
    gdn_qkv_raw: Vec<u8>,
    gdn_gate_raw: Vec<u8>,
    gdn_alpha_raw: Vec<u8>,
    gdn_beta_raw: Vec<u8>,
    gdn_ssm_out_raw: Vec<u8>,
    conv_state: Vec<f32>,
    conv_kernel: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    delta_state: Vec<f32>,
    ssm_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    router_w: Vec<f32>,
    gate_all: Vec<u8>,
    up_all: Vec<u8>,
    down_q5_all: Vec<u8>,
    down_q6_all: Vec<u8>,
    shared_input_scale: Vec<f32>,
    shared_q8_gate: Vec<u8>,
    shared_q8_up: Vec<u8>,
    shared_q8_down: Vec<u8>,
    shared_q4_gate: Vec<u8>,
    shared_q4_up: Vec<u8>,
    shared_q6_down: Vec<u8>,
}

#[cfg(target_os = "macos")]
impl QwenMoeLlamaPrefillLayerChainFixture {
    pub const SEQ_LEN: usize = 2;
    pub const HIDDEN_DIM: usize = 256;
    pub const HEAD_DIM: usize = 256;
    pub const KV_DIM: usize = 256;
    pub const D_INNER: usize = 256;
    pub const D_STATE: usize = 2;
    pub const N_GROUP: usize = 1;
    pub const DT_RANK: usize = 2;
    pub const CONV_KERNEL_SIZE: usize = 4;
    pub const FFN_DIM: usize = 256;
    pub const N_EXPERT: usize = 4;
    pub const N_EXPERT_USED: usize = 2;

    const Q_DIM: usize = 256;
    const CONV_CHANNELS: usize = Self::D_INNER + 2 * Self::N_GROUP * Self::D_STATE;

    pub fn attention_spec(&self, layer_idx: usize) -> crate::QwenPrefillChainSpecRef<'_> {
        let q_weight = crate::PrefillAtnCoreWeightView {
            raw: &self.attn_q_raw,
            quant: crate::TensoropsQuant::Q4K,
            rows: 2 * Self::Q_DIM,
            cols: Self::HIDDEN_DIM,
        };
        let k_weight = crate::PrefillAtnCoreWeightView {
            raw: &self.attn_k_raw,
            quant: crate::TensoropsQuant::Q4K,
            rows: Self::KV_DIM,
            cols: Self::HIDDEN_DIM,
        };
        let v_weight = crate::PrefillAtnCoreWeightView {
            raw: &self.attn_v_raw,
            quant: crate::TensoropsQuant::Q4K,
            rows: Self::KV_DIM,
            cols: Self::HIDDEN_DIM,
        };
        let o_weight = crate::PrefillAtnCoreWeightView {
            raw: &self.attn_o_raw,
            quant: crate::TensoropsQuant::Q4K,
            rows: Self::HIDDEN_DIM,
            cols: Self::Q_DIM,
        };
        crate::QwenPrefillChainSpecRef::Attention {
            layer_idx,
            core: crate::PrefillAtnOTailBackendSpecRef {
                core: crate::PrefillAtnCoreBackendSpecRef {
                    attn_norm_w: &self.attn_norm,
                    q_norm_w: &self.q_norm,
                    k_norm_w: &self.k_norm,
                    q_weight,
                    k_weight,
                    v_weight,
                    seq_len: Self::SEQ_LEN,
                    num_heads: 1,
                    num_kv_heads: 1,
                    head_dim: Self::HEAD_DIM,
                    hidden_dim: Self::HIDDEN_DIM,
                    q_dim: Self::Q_DIM,
                    kv_dim: Self::KV_DIM,
                    n_rot: 64,
                    rope_theta: 1_000_000.0,
                    scale: 1.0 / (Self::HEAD_DIM as f32).sqrt(),
                    norm_eps: 1e-6,
                    pos_start: 0,
                },
                o_weight,
            },
            moe: self.moe_spec(false),
        }
    }

    fn gdn_quant(raw: &[u8], rows: usize, cols: usize) -> crate::GdnBackendWeightRef<'_> {
        crate::GdnBackendWeightRef::Quant(crate::PrefillAtnCoreWeightView {
            raw,
            quant: crate::TensoropsQuant::Q4K,
            rows,
            cols,
        })
    }

    pub fn gdn_spec(&self, layer_idx: usize) -> crate::QwenPrefillChainSpecRef<'_> {
        crate::QwenPrefillChainSpecRef::Gdn {
            layer_idx,
            layer: crate::QwenPrefillGdnBackendSpecRef {
                seq_len: Self::SEQ_LEN,
                hidden_dim: Self::HIDDEN_DIM,
                d_inner: Self::D_INNER,
                d_state: Self::D_STATE,
                n_group: Self::N_GROUP,
                dt_rank: Self::DT_RANK,
                conv_kernel_size: Self::CONV_KERNEL_SIZE,
                attn_norm_w: &self.attn_norm,
                qkv_weight: Self::gdn_quant(
                    &self.gdn_qkv_raw,
                    Self::CONV_CHANNELS,
                    Self::HIDDEN_DIM,
                ),
                gate_weight: Self::gdn_quant(&self.gdn_gate_raw, Self::D_INNER, Self::HIDDEN_DIM),
                alpha_weight: Self::gdn_quant(&self.gdn_alpha_raw, Self::DT_RANK, Self::HIDDEN_DIM),
                beta_weight: Self::gdn_quant(&self.gdn_beta_raw, Self::DT_RANK, Self::HIDDEN_DIM),
                conv_state: &self.conv_state,
                conv_kernel: &self.conv_kernel,
                dt_bias: &self.dt_bias,
                ssm_a: &self.ssm_a,
                delta_state: &self.delta_state,
                ssm_norm: &self.ssm_norm,
                ssm_out_weight: Self::gdn_quant(
                    &self.gdn_ssm_out_raw,
                    Self::HIDDEN_DIM,
                    Self::D_INNER,
                ),
                post_attn_norm_w: &self.ffn_norm,
                norm_eps: 1e-6,
            },
            moe: self.moe_spec(true),
        }
    }

    pub fn contiguous_specs(
        &self,
        first_layer_idx: usize,
        layer_count: usize,
    ) -> Vec<crate::QwenPrefillChainSpecRef<'_>> {
        (0..layer_count)
            .map(|offset| {
                let layer_idx = first_layer_idx + offset;
                if offset % 2 == 0 {
                    self.attention_spec(layer_idx)
                } else {
                    self.gdn_spec(layer_idx)
                }
            })
            .collect()
    }

    pub const fn expected_kv_len() -> usize {
        Self::SEQ_LEN * Self::KV_DIM
    }

    pub const fn expected_conv_state_len() -> usize {
        (Self::CONV_KERNEL_SIZE - 1) * Self::CONV_CHANNELS
    }

    pub const fn expected_delta_state_len() -> usize {
        Self::D_INNER * Self::D_STATE
    }

    fn moe_spec(&self, q5_down: bool) -> crate::QwenMoePrefillBackendSpecRef<'_> {
        let sparse_quant = crate::QwenMoeLlamaIdQuantSet {
            gate: crate::QwenMoeLlamaIdQuant::Q4K,
            up: crate::QwenMoeLlamaIdQuant::Q4K,
            down: if q5_down {
                crate::QwenMoeLlamaIdQuant::Q5K
            } else {
                crate::QwenMoeLlamaIdQuant::Q6K
            },
        };
        let (down_all, down_expert_bytes, shared_quant, shared_gate, shared_up, shared_down) =
            if q5_down {
                (
                    self.down_q5_all.as_slice(),
                    Self::HIDDEN_DIM * 176,
                    crate::QwenMoeLlamaIdQuantSet {
                        gate: crate::QwenMoeLlamaIdQuant::Q8Zero,
                        up: crate::QwenMoeLlamaIdQuant::Q8Zero,
                        down: crate::QwenMoeLlamaIdQuant::Q8Zero,
                    },
                    self.shared_q8_gate.as_slice(),
                    self.shared_q8_up.as_slice(),
                    self.shared_q8_down.as_slice(),
                )
            } else {
                (
                    self.down_q6_all.as_slice(),
                    Self::HIDDEN_DIM * 210,
                    crate::QwenMoeLlamaIdQuantSet {
                        gate: crate::QwenMoeLlamaIdQuant::Q4K,
                        up: crate::QwenMoeLlamaIdQuant::Q4K,
                        down: crate::QwenMoeLlamaIdQuant::Q6K,
                    },
                    self.shared_q4_gate.as_slice(),
                    self.shared_q4_up.as_slice(),
                    self.shared_q6_down.as_slice(),
                )
            };
        crate::QwenMoePrefillBackendSpecRef {
            ffn_norm_w: &self.ffn_norm,
            norm_eps: 3.0e-5,
            router_w: &self.router_w,
            gate_all: &self.gate_all,
            up_all: &self.up_all,
            down_all,
            gate_expert_bytes: Self::FFN_DIM * 144,
            up_expert_bytes: Self::FFN_DIM * 144,
            down_expert_bytes,
            shared_input_scale: &self.shared_input_scale,
            shared_gate,
            shared_up,
            shared_down,
            sparse_quant,
            shared_quant,
            route_algorithm: crate::QwenRouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1,
            n_expert: Self::N_EXPERT,
            n_expert_used: Self::N_EXPERT_USED,
            hidden_dim: Self::HIDDEN_DIM,
            ffn_dim: Self::FFN_DIM,
        }
    }
}

#[cfg(target_os = "macos")]
fn scaled_q4k_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let blocks = rows * (cols / 256);
    let mut raw = Vec::with_capacity(blocks * 144);
    for block_idx in 0..blocks {
        let mut block = q4k_block_fixed();
        let scale = 0.000_122_070_31 * (1 + (seed + block_idx) % 4) as f32;
        block[0..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(scale * 0.25).to_le_bytes());
        raw.extend_from_slice(&block);
    }
    raw
}

#[cfg(target_os = "macos")]
fn scaled_q5k_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let blocks = rows * (cols / 256);
    let mut raw = Vec::with_capacity(blocks * 176);
    for block_idx in 0..blocks {
        let mut block = q5k_block_fixed();
        let scale = 0.000_061_035_156 * (1 + (seed + block_idx) % 4) as f32;
        block[0..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(scale * 0.25).to_le_bytes());
        raw.extend_from_slice(&block);
    }
    raw
}

#[cfg(target_os = "macos")]
fn scaled_q6k_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let blocks = rows * (cols / 256);
    let mut raw = Vec::with_capacity(blocks * 210);
    for block_idx in 0..blocks {
        let mut block = q6k_block_fixed();
        let scale = 0.000_030_517_578 * (1 + (seed + block_idx) % 4) as f32;
        block[208..210].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        raw.extend_from_slice(&block);
    }
    raw
}

#[cfg(target_os = "macos")]
pub fn scaled_q8_0_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let blocks = rows * (cols / 32);
    let mut raw = Vec::with_capacity(blocks * 34);
    for block_idx in 0..blocks {
        let mut block = [0u8; 34];
        let scale = 0.000_061_035_156 * (1 + (seed + block_idx) % 4) as f32;
        block[0..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        for lane in 0..32 {
            let value = ((seed + block_idx * 11 + lane * 7) % 31) as i8 - 15;
            block[2 + lane] = if value == 0 { 1 } else { value } as u8;
        }
        raw.extend_from_slice(&block);
    }
    raw
}

#[cfg(target_os = "macos")]
pub fn qwen_moe_llama_prefill_layer_chain_fixture() -> QwenMoeLlamaPrefillLayerChainFixture {
    let hidden = (0..QwenMoeLlamaPrefillLayerChainFixture::SEQ_LEN
        * QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM)
        .map(|index| ((index * 13 % 41) as f32 - 20.0) * 0.001_953_125)
        .collect();
    let norm = (0..QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM)
        .map(|index| 0.875 + (index % 5) as f32 * 0.031_25)
        .collect::<Vec<_>>();
    let router_w = (0..QwenMoeLlamaPrefillLayerChainFixture::N_EXPERT
        * QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM)
        .map(|index| {
            let expert = index / QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM;
            let col = index % QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM;
            (((expert * 17 + col * 13) % 19) as f32 - 9.0) * 0.000_122_070_31
        })
        .collect();
    let shared_input_scale = (0..QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM)
        .map(|index| ((index * 7 % 17) as f32 - 8.0) * 0.000_122_070_31)
        .collect();
    let conv_state = (0..QwenMoeLlamaPrefillLayerChainFixture::expected_conv_state_len())
        .map(|index| ((index % 23) as f32 - 11.0) * 0.000_976_562_5)
        .collect();
    let delta_state = (0..QwenMoeLlamaPrefillLayerChainFixture::expected_delta_state_len())
        .map(|index| ((index % 29) as f32 - 14.0) * 0.000_488_281_25)
        .collect();
    let conv_kernel = (0..QwenMoeLlamaPrefillLayerChainFixture::CONV_KERNEL_SIZE
        * QwenMoeLlamaPrefillLayerChainFixture::CONV_CHANNELS)
        .map(|index| ((index * 5 % 13) as f32 - 6.0) * 0.000_244_140_63)
        .collect();
    let gate_all = scaled_q4k_matrix(
        QwenMoeLlamaPrefillLayerChainFixture::N_EXPERT
            * QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
        QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
        101,
    );
    let up_all = scaled_q4k_matrix(
        QwenMoeLlamaPrefillLayerChainFixture::N_EXPERT
            * QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
        QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
        211,
    );

    QwenMoeLlamaPrefillLayerChainFixture {
        hidden,
        attn_norm: norm.clone(),
        q_norm: vec![1.0; QwenMoeLlamaPrefillLayerChainFixture::HEAD_DIM],
        k_norm: vec![1.0; QwenMoeLlamaPrefillLayerChainFixture::HEAD_DIM],
        attn_q_raw: scaled_q4k_matrix(
            2 * QwenMoeLlamaPrefillLayerChainFixture::Q_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            3,
        ),
        attn_k_raw: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::KV_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            7,
        ),
        attn_v_raw: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::KV_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            11,
        ),
        attn_o_raw: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::Q_DIM,
            17,
        ),
        gdn_qkv_raw: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::CONV_CHANNELS,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            23,
        ),
        gdn_gate_raw: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::D_INNER,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            29,
        ),
        gdn_alpha_raw: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::DT_RANK,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            31,
        ),
        gdn_beta_raw: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::DT_RANK,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            37,
        ),
        gdn_ssm_out_raw: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::D_INNER,
            41,
        ),
        conv_state,
        conv_kernel,
        dt_bias: vec![-0.25, -0.125],
        ssm_a: vec![-0.5, -0.375],
        delta_state,
        ssm_norm: vec![
            1.0;
            QwenMoeLlamaPrefillLayerChainFixture::D_INNER
                / QwenMoeLlamaPrefillLayerChainFixture::DT_RANK
        ],
        ffn_norm: norm,
        router_w,
        gate_all,
        up_all,
        down_q5_all: scaled_q5k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::N_EXPERT
                * QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
            307,
        ),
        down_q6_all: scaled_q6k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::N_EXPERT
                * QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
            401,
        ),
        shared_input_scale,
        shared_q8_gate: scaled_q8_0_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            503,
        ),
        shared_q8_up: scaled_q8_0_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            601,
        ),
        shared_q8_down: scaled_q8_0_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
            701,
        ),
        shared_q4_gate: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            809,
        ),
        shared_q4_up: scaled_q4k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            907,
        ),
        shared_q6_down: scaled_q6k_matrix(
            QwenMoeLlamaPrefillLayerChainFixture::HIDDEN_DIM,
            QwenMoeLlamaPrefillLayerChainFixture::FFN_DIM,
            1_009,
        ),
    }
}

/// Single-token decode attention fixture: 결정적 q/k_cache/v_cache + CPU reference.
///
/// ground truth = production default `attention_decode_flash`
/// (f16 accumulator + branched online softmax). MSL `attn_decode` 커널 검증용 —
/// 직접 재구현 금지, 검증된 CPU 경로를 reference 로 쓴다.
/// GQA: num_heads=2, num_kv_heads=1(heads_per_group=2), head_dim=128, kv_len=3.
pub struct AttnDecodeFixture {
    pub q: Vec<f32>,
    pub k_cache: Vec<u16>,
    pub v_cache: Vec<u16>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_len: usize,
    pub scale: f32,
    pub reference: Vec<f32>,
}

pub fn attn_decode_fixture() -> AttnDecodeFixture {
    attn_decode_fixture_shaped(2, 1, 128, 3)
}

/// 임의 shape 의 attention decode fixture(GQA factor·head_dim·kv_len 검증용).
pub fn attn_decode_fixture_shaped(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> AttnDecodeFixture {
    use rnb_cpu::kernels::attention::attention_decode_flash;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let kv_dim = num_kv_heads * head_dim;

    // f16-정확 값으로 채워 round-trip 무손실(exp 외 산술 일치 최대화).
    let mut q = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        for d in 0..head_dim {
            q[h * head_dim + d] = (((h + d) % 5) as f32) * 0.125; // 0,0.125,..,0.5
        }
    }
    let mut k_cache = vec![0u16; kv_len * kv_dim];
    let mut v_cache = vec![0u16; kv_len * kv_dim];
    for j in 0..kv_len {
        for d in 0..kv_dim {
            let kv = (((j + d) % 4) as f32) * 0.25; // 0,0.25,0.5,0.75
            let vv = (((j * 2 + d) % 3) as f32) * 0.5; // 0,0.5,1.0
            k_cache[j * kv_dim + d] = half::f16::from_f32(kv).to_bits();
            v_cache[j * kv_dim + d] = half::f16::from_f32(vv).to_bits();
        }
    }

    let mut reference = vec![0.0f32; num_heads * head_dim];
    attention_decode_flash(
        &q,
        &k_cache,
        &v_cache,
        &mut reference,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        scale,
        None,
        None,
    );

    AttnDecodeFixture {
        q,
        k_cache,
        v_cache,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        scale,
        reference,
    }
}

/// Text M-RoPE fixture: 결정적 입력 + CPU `rope_mrope_text_inplace` reference.
/// MSL `rope_mrope` 커널 검증용. (head_dim=128, dim=256(2 head), n_rot=128).
pub struct RopeMropeFixture {
    pub data: Vec<f32>,
    pub reference: Vec<f32>,
    pub head_dim: usize,
    pub dim: usize,
    pub n_rot: usize,
    pub theta: f32,
    pub pos: usize,
}

pub fn rope_mrope_fixture() -> RopeMropeFixture {
    use rnb_cpu::kernels::rope::rope_mrope_text_inplace;

    let head_dim = 128usize;
    let dim = 256usize; // 2 heads
    let n_rot = 128usize;
    let theta = 1.0e6f32;
    let pos = 5usize;

    let mut data = vec![0.0f32; dim];
    for (i, x) in data.iter_mut().enumerate() {
        *x = (((i % 11) as f32) - 5.0) * 0.1; // -0.5..0.5
    }

    let mut reference = data.clone();
    rope_mrope_text_inplace(&mut reference, pos, head_dim, dim, n_rot, theta);

    RopeMropeFixture {
        data,
        reference,
        head_dim,
        dim,
        n_rot,
        theta,
        pos,
    }
}

/// Per-head q/k RMSNorm fixture: 결정적 입력 + CPU `rms_norm_into` reference.
/// MSL `qk_norm` 커널 검증용. weight(head_dim)는 모든 head 공유, head 별로
/// head_dim 슬라이스를 각각 RMSNorm 한다 (표준 Qwen3 attn q_norm/k_norm).
/// num_heads=4, head_dim=128(9B head_dim).
pub struct QkNormFixture {
    pub data: Vec<f32>,
    pub weight: Vec<f32>,
    pub reference: Vec<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
}

pub fn qk_norm_fixture() -> QkNormFixture {
    qk_norm_fixture_shaped(4, 128)
}

/// 임의 shape 의 q/k norm fixture(num_heads·head_dim 검증용).
pub fn qk_norm_fixture_shaped(num_heads: usize, head_dim: usize) -> QkNormFixture {
    use rnb_cpu::kernels::norm::rms_norm_into;

    let eps = 1.0e-6f32;
    let mut data = vec![0.0f32; num_heads * head_dim];
    for (i, x) in data.iter_mut().enumerate() {
        *x = (((i % 7) as f32) - 3.0) * 0.1; // -0.3..0.3
    }
    let mut weight = vec![0.0f32; head_dim];
    for (i, w) in weight.iter_mut().enumerate() {
        *w = 0.5 + ((i % 5) as f32) * 0.1; // 0.5..0.9
    }

    let mut reference = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let off = h * head_dim;
        rms_norm_into(
            &data[off..off + head_dim],
            &weight,
            eps,
            &mut reference[off..off + head_dim],
        );
    }

    QkNormFixture {
        data,
        weight,
        reference,
        num_heads,
        head_dim,
        eps,
    }
}

/// GDN decode(seq_len=1) depthwise causal conv1d + SiLU fixture.
/// MSL `ssm_conv1d_silu` 커널 검증용. CPU `ssm_conv1d_silu_into`(seq_len=1) reference.
/// input/weight = [kernel_size*channels] flat, reference = [channels].
pub struct SsmConvSiluFixture {
    pub input: Vec<f32>,
    pub weight: Vec<f32>,
    pub reference: Vec<f32>,
}

/// 임의 shape(channels·kernel_size) 의 conv1d+silu fixture.
pub fn ssm_conv1d_silu_fixture_shaped(channels: usize, kernel_size: usize) -> SsmConvSiluFixture {
    use rnb_cpu::kernels::conv::ssm_conv1d_silu_into;

    let mut input = vec![0.0f32; kernel_size * channels];
    for (i, x) in input.iter_mut().enumerate() {
        *x = (((i % 11) as f32) - 5.0) * 0.05; // -0.25..0.25
    }
    let mut weight = vec![0.0f32; kernel_size * channels];
    for (i, w) in weight.iter_mut().enumerate() {
        *w = 0.1 + ((i % 7) as f32) * 0.05; // 0.1..0.4
    }

    let mut reference = vec![0.0f32; channels];
    ssm_conv1d_silu_into(&input, &weight, &mut reference, 1, channels, kernel_size);

    SsmConvSiluFixture {
        input,
        weight,
        reference,
    }
}

/// GDN delta_net recurrent scan 1-step(decode, seq_len=1) fixture.
/// MSL `delta_net_step` 커널 검증용. CPU `delta_net_scan_into`(seq_len=1) reference.
/// q/k=[num_heads*head_k_dim], v=[num_heads*head_v_dim], gate/beta=[num_heads],
/// state=[num_heads*head_v_dim*head_k_dim]. ref_out + ref_state(갱신 후) 둘 다.
pub struct DeltaNetStepFixture {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub beta: Vec<f32>,
    pub state_in: Vec<f32>,
    pub ref_out: Vec<f32>,
    pub ref_state: Vec<f32>,
}

pub fn delta_net_step_fixture_shaped(
    num_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> DeltaNetStepFixture {
    use rnb_cpu::kernels::delta_net::delta_net_scan_into;

    // q/k 는 k-head(num_k_heads)만 — GQA 면 v-head 가 공유.
    let q: Vec<f32> = (0..num_k_heads * head_k_dim)
        .map(|i| (((i % 13) as f32) - 6.0) * 0.04) // -0.24..0.24
        .collect();
    let k: Vec<f32> = (0..num_k_heads * head_k_dim)
        .map(|i| (((i % 9) as f32) - 4.0) * 0.05) // -0.2..0.2
        .collect();
    let v: Vec<f32> = (0..num_heads * head_v_dim)
        .map(|i| (((i % 7) as f32) - 3.0) * 0.06) // -0.18..0.18
        .collect();
    // gate(=alpha-gate): exp(gate) 가 decay → 음수 영역(decay<1)으로.
    let gate: Vec<f32> = (0..num_heads)
        .map(|h| -0.1 - (h % 4) as f32 * 0.05)
        .collect();
    let beta: Vec<f32> = (0..num_heads)
        .map(|h| 0.3 + (h % 5) as f32 * 0.08)
        .collect();
    let state_in: Vec<f32> = (0..num_heads * head_v_dim * head_k_dim)
        .map(|i| (((i % 11) as f32) - 5.0) * 0.02) // -0.1..0.1
        .collect();

    // CPU reference 는 q/k 를 v-head 로 repeat 한 뒤 delta_net_scan_into (CPU 와 동일 GQA).
    let mut q_rep = vec![0.0f32; num_heads * head_k_dim];
    let mut k_rep = vec![0.0f32; num_heads * head_k_dim];
    for vh in 0..num_heads {
        let kh = vh % num_k_heads;
        q_rep[vh * head_k_dim..vh * head_k_dim + head_k_dim]
            .copy_from_slice(&q[kh * head_k_dim..kh * head_k_dim + head_k_dim]);
        k_rep[vh * head_k_dim..vh * head_k_dim + head_k_dim]
            .copy_from_slice(&k[kh * head_k_dim..kh * head_k_dim + head_k_dim]);
    }
    let mut ref_state = state_in.clone();
    let mut ref_out = vec![0.0f32; num_heads * head_v_dim];
    delta_net_scan_into(
        &q_rep,
        &k_rep,
        &v,
        &gate,
        &beta,
        &mut ref_state,
        &mut ref_out,
        1,
        num_heads,
        head_k_dim,
        head_v_dim,
    );

    DeltaNetStepFixture {
        q,
        k,
        v,
        gate,
        beta,
        state_in,
        ref_out,
        ref_state,
    }
}

/// GDN delta_net chunkwise parallel scan(prefill, seq_len>1) fixture.
/// MSL `delta_net_scan_chunk` 커널 검증용. CPU `delta_net_scan_chunkwise`(M1 oracle) reference.
/// GQA 는 caller(prefill) 가 이미 repeat 푼 q/k(num_heads) 를 넘기므로 fixture 도 num_heads 단일.
/// q/k=[seq*num_heads*head_k_dim], v=[seq*num_heads*head_v_dim], gate/beta=[seq*num_heads],
/// state=[num_heads*head_v_dim*head_k_dim]. ref_out=[seq*num_heads*head_v_dim] + ref_state(hand-off 후).
pub struct DeltaNetScanChunkFixture {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub beta: Vec<f32>,
    pub state_in: Vec<f32>,
    pub ref_out: Vec<f32>,
    pub ref_state: Vec<f32>,
}

pub fn delta_net_scan_chunk_fixture_shaped(
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
) -> DeltaNetScanChunkFixture {
    use rnb_cpu::kernels::delta_net::delta_net_scan_chunkwise;

    // delta_net_step_fixture_shaped 와 같은 deterministic 생성식(token 축 확장).
    let q: Vec<f32> = (0..seq_len * num_heads * head_k_dim)
        .map(|i| (((i % 13) as f32) - 6.0) * 0.04) // -0.24..0.24
        .collect();
    let k: Vec<f32> = (0..seq_len * num_heads * head_k_dim)
        .map(|i| (((i % 9) as f32) - 4.0) * 0.05) // -0.2..0.2
        .collect();
    let v: Vec<f32> = (0..seq_len * num_heads * head_v_dim)
        .map(|i| (((i % 7) as f32) - 3.0) * 0.06) // -0.18..0.18
        .collect();
    // gate(=alpha-gate): exp(gate) 가 decay → 음수 영역(decay<1).
    let gate: Vec<f32> = (0..seq_len * num_heads)
        .map(|i| -0.1 - (i % 4) as f32 * 0.05)
        .collect();
    let beta: Vec<f32> = (0..seq_len * num_heads)
        .map(|i| 0.3 + (i % 5) as f32 * 0.08)
        .collect();
    let state_in: Vec<f32> = (0..num_heads * head_v_dim * head_k_dim)
        .map(|i| (((i % 11) as f32) - 5.0) * 0.02) // -0.1..0.1
        .collect();

    let mut ref_state = state_in.clone();
    let ref_out = delta_net_scan_chunkwise(
        &q,
        &k,
        &v,
        &gate,
        &beta,
        &mut ref_state,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
        chunk_size,
    );

    DeltaNetScanChunkFixture {
        q,
        k,
        v,
        gate,
        beta,
        state_in,
        ref_out,
        ref_state,
    }
}
