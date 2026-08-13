use std::sync::OnceLock;

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

pub fn expanded_weight_cache_allowed() -> bool {
    env_bool("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", false)
}

fn expanded_env_bool(name: &str, default: bool) -> bool {
    expanded_weight_cache_allowed() && env_bool(name, default)
}

fn expanded_env_force(name: &str) -> bool {
    expanded_weight_cache_allowed()
        && std::env::var(name)
            .ok()
            .map(|value| value.eq_ignore_ascii_case("force"))
            .unwrap_or(false)
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

pub fn output_logits_enabled() -> bool {
    env_bool("RNB_CUDA_OUTPUT_LOGITS", true)
}

pub fn output_argmax_enabled() -> bool {
    env_bool("RNB_CUDA_OUTPUT_ARGMAX", false)
}

pub fn q6k_output_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_OUTPUT_WARP8", true)
}

pub fn q6k_fused_argmax_gpu_reduce_enabled(rows: usize) -> bool {
    env_bool("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE", rows >= 8192)
}
pub fn q8_0_gemv_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_Q8_0_GEMV_WARP4", true)
}

pub fn q8_0_gemv_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q8_0_GEMV_WARP8", true)
}

pub fn q4k_gemv_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_GEMV_WARP8", true)
}

pub fn q4k_packed_gemv_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_PACKED_GEMV_WARP4", false)
}

pub fn q6k_packed_gemv_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_PACKED_GEMV_WARP4", false)
}

pub fn q6k_packed_batch_warp4_enabled(blocks_per_row: usize) -> bool {
    blocks_per_row >= 8 && env_bool("RNB_CUDA_Q6_PACKED_BATCH_WARP4", true)
}

pub fn q6k_packed_batch_seq8_enabled(seq_len: usize, blocks_per_row: usize) -> bool {
    let _ = (seq_len, blocks_per_row);
    env_bool("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8", false)
}

pub fn q8_0_output_q8dot_argmax_enabled() -> bool {
    env_bool("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX", false)
}

pub fn dense_expert_graph_enabled() -> bool {
    env_bool("RNB_CUDA_DENSE_EXPERT_GRAPH", false)
}

pub fn cu69_dense_chain_graph_enabled() -> bool {
    env_is_one("RNB_CU69_DENSE_CHAIN_GRAPH")
}

pub fn cu69_dense_chain_graph_trace_enabled() -> bool {
    env_is_one("RNB_CU69_DENSE_CHAIN_GRAPH_TRACE")
}

pub fn cu71_layer_segment_graph_enabled() -> bool {
    env_is_one("RNB_CU71_LAYER_SEGMENT_GRAPH")
}

/// cu74: persistent cooperative decode kernel for Gemma4 E2B.
/// Opt-in only; eager dispatch remains default until token-by-token
/// correctness and ABAB gates pass.
pub fn persistent_decode_enabled() -> bool {
    env_is_one("RNB_CUDA_PERSISTENT_DECODE")
}

pub fn cu71_layer_segment_graph_trace_enabled() -> bool {
    env_is_one("RNB_CU71_LAYER_SEGMENT_GRAPH_TRACE")
}

pub fn qwen35_decode_moe_graph_enabled() -> bool {
    env_bool("RNB_CUDA_MOE_GRAPH", true)
}

pub fn qwen35_selected_sparse_compound_graph_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH", true)
}

pub fn qwen35_selected_sparse_compound_graph_zero_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO", false)
}

pub fn q4k_gemv_batch_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_GEMV_BATCH_WARP8", true)
}

pub fn q4k_batch_raw_seq4_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    // cu262: seq4 keeps each token's raw-F32 accumulation order while sharing
    // one weight decode across four tokens. Keep the original wide-row gate,
    // and admit narrow projections only when at least a 64-token slab supplies
    // enough independent row/sequence CTAs to offset the 4x smaller grid.
    let enough_parallel_work = rows >= 1024 || (rows >= 64 && seq_len >= 64);
    let default = seq_len >= 8 && enough_parallel_work && blocks_per_row >= 4;
    env_bool("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", default)
}

/// cu219: MMQ tile32 의 최소 seq 게이트 (Q4_K/Q6_K). 커널은 partial tile 을
/// token bounds 마스킹으로 완전 지원하므로 32 는 정확성 경계가 아니라
/// 휴리스틱이었다. RTX 3090 27B 15/100 에서 8 로 낮추면 host-input 위임과
/// 결합해 generation −14.5%(3/3), 1139-token 도 −37.6%로 f32/구세대 대비
/// 우세 구간만 있었다. `RNB_CUDA_MMQ_TILE32_MIN_SEQ` 로 조절한다.
pub fn mmq_tile32_min_seq() -> usize {
    std::env::var("RNB_CUDA_MMQ_TILE32_MIN_SEQ")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(8)
}

pub fn q4k_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= mmq_tile32_min_seq() && rows >= 1024 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q4K_MMQ_TILE32", true)
}

/// cu226: MMQ tile 의 CTA 당 seq 폭 64 확대 게이트. ncu 재귀속에서 tile32 가
/// cu223 융합 후에도 LSU-bound 였고, grid.y 의 32-seq slab 들이 같은 weight
/// tile 을 반복 로드/unpack 하는 것이 a-side 명령·섹터의 지배 항이었다.
/// 64 폭은 a-tile 비용을 출력당 절반으로 상각한다 (per-element 누산 순서는
/// tile32 와 동일 — bitwise 계약 유지). partial tile 은 커널이 token bounds
/// 마스킹으로 지원하지만 seq < 64 는 b-slab 절반이 항상 유휴라 tile32 가
/// 남는다. 진단 대조는 `RNB_CUDA_MMQ_TILE_SEQ64=0`.
pub fn mmq_tile_seq64_enabled(seq_len: usize) -> bool {
    seq_len >= 64 && env_bool("RNB_CUDA_MMQ_TILE_SEQ64", true)
}

/// cu228: Q4_K 64x64 tile 게이트 — seq64 tile 의 b-side(activation) 로드가
/// loader ops 의 2/3 이고 grid.x 의 32-row CTA 마다 재발행되므로, row 64
/// tile 이 b 를 출력당 절반으로 상각한다. 512-thread CTA 가 32x64 와 같은
/// per-thread accumulator 레이아웃을 유지해 bitwise 계약 불변. rows >= 64
/// 는 tile 높이의 구조적 최소치다. 진단 대조는 `RNB_CUDA_Q4K_MMQ_TILE64=0`.
pub fn q4k_mmq_tile64_enabled(seq_len: usize, rows: usize) -> bool {
    mmq_tile_seq64_enabled(seq_len) && rows >= 64 && env_bool("RNB_CUDA_Q4K_MMQ_TILE64", true)
}

/// cu228: Q5_K/Q6_K 64x64 tile 게이트 — Q4 와 같은 b-side 상각 (rows >= 64
/// 는 tile 높이의 구조적 최소치). 진단 대조는 각 env `=0`.
pub fn q5k_mmq_tile64_enabled(seq_len: usize, rows: usize) -> bool {
    mmq_tile_seq64_enabled(seq_len) && rows >= 64 && env_bool("RNB_CUDA_Q5K_MMQ_TILE64", true)
}
pub fn q6k_mmq_tile64_enabled(seq_len: usize, rows: usize) -> bool {
    mmq_tile_seq64_enabled(seq_len) && rows >= 64 && env_bool("RNB_CUDA_Q6K_MMQ_TILE64", true)
}

pub fn q2k_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= 32 && rows >= 1024 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q2K_MMQ_TILE32", true)
}
pub fn q3k_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= 32 && rows >= 1024 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q3K_MMQ_TILE32", true)
}
pub fn q6k_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= mmq_tile32_min_seq() && rows >= 1024 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q6K_MMQ_TILE32", true)
}
/// cu222: Q5_K MMQ tile32 게이트 — Q4/Q6 과 같은 min_seq/shape 조건.
pub fn q5k_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= mmq_tile32_min_seq() && rows >= 1024 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q5K_MMQ_TILE32", true)
}

/// cu221: dev-input dense FFN 의 Q6_K down q8dot 분기를 MMQ tile32 로
/// 라우팅하는 게이트 (Q4 down 은 q4k_batch_q8dot_to_dev 내부 MMQ 라우팅이
/// 이미 있었다). min_seq(기본 8) 미달인 verify/decode 는 이 게이트와
/// 무관하게 기존 packed/paired 경로를 유지한다. `=0` 은 진단 opt-out.
pub fn q6k_down_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    q6k_mmq_tile32_enabled(seq_len, rows, blocks_per_row)
        && env_bool("RNB_CUDA_Q6K_DOWN_MMQ_TILE32", true)
}

/// cu221: dense Qwen(Qwen35) attention 층을 prefill device carrier chain 에
/// 연결하는 게이트 — attention device-input + dense SwiGLU FFN carrier.
/// `=0` 은 attention 층을 기존 host materialize 경로로 되돌린다.
pub fn qwen_dense_prefill_attention_device_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN_DENSE_PREFILL_ATTENTION_DEVICE", true)
}

/// cu219: 위임된 host-input prefill 배치에서 Q5_K 를 raw F32 대신 q8dot
/// batch(wide) 세대로 태우는 게이트. Q5 는 MMQ tile32 가 없어 위임 후에도
/// raw 로 남았었다 (27B 15-token prefill 의 102ms/48calls). verify 의
/// `RNB_CUDA_Q5K_BATCH_Q8DOT`(2..=4 밴드) 정책 소유권은 건드리지 않는다.
/// `RNB_CUDA_PREFILL_Q5_BATCH_Q8DOT=0` 은 raw 로 되돌리는 진단 opt-out.
pub fn prefill_q5_batch_q8dot_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    seq_len >= 2
        && rows >= 1024
        && blocks_per_row >= 4
        && env_bool("RNB_CUDA_PREFILL_Q5_BATCH_Q8DOT", true)
}

/// cu219: host-input prefill 배치(gemv_batch)를 dev-input 경로의 검증된 kernel
/// 우선순위(MMQ/MMA v3/pair2/q8dot)로 위임한다. 기존 host 분기는 seq4/raw
/// 세대에 갇혀 15-token prefill 이 층당 weight 를 토큰 수만큼 재읽었다.
/// `RNB_CUDA_PREFILL_BATCH_DEV_DISPATCH=0` 은 기존 host 분기 opt-out.
pub fn prefill_batch_dev_dispatch_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_BATCH_DEV_DISPATCH", true)
}
pub fn q8_0_mmq_tile32_enabled(seq_len: usize, rows: usize, blocks_per_row: usize) -> bool {
    let eligible = seq_len >= 32 && rows >= 128 && blocks_per_row >= 4;
    eligible && env_bool("RNB_CUDA_Q8_0_MMQ_TILE32", true)
}

/// `seq_len == 2` 전용 Q4_K gate/up q8dot 배치 커널.
///
/// mt103 실측으로 기본값을 내렸다. RTX 3090 Qwen3.6 27B MTP Q4_K_M의 `k=1` MTP device
/// verify(2-position window)에서 이 커널이 verify kernel을 `89.1ms`에서 `1933.1ms`로
/// 21배 늘렸고, 제품 처리량은 `14.0` 대 `0.9 tok/s`였다. 4-position window는 이 분기를
/// 타지 않아 영향이 없었고, 같은 토글을 Qwen3.6 35B-A3B(90.9 대 90.9)와 Gemma 4
/// 26B-A4B(warm 교차 25.0/24.9/24.7/24.7)에서 켜고 꺼도 차이가 없어 이 경로를 타는
/// 워크로드는 27B dense verify뿐이다. 즉 어느 측정에서도 이득이 확인되지 않는다.
/// 필요하면 `RNB_CUDA_Q4K_GATE_UP_BATCH_SEQ2_Q8DOT=1`로 되켤 수 있다.
pub fn q4k_gate_up_batch_seq2_q8dot_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_GATE_UP_BATCH_SEQ2_Q8DOT", false)
}

/// Q4_K dense gate/up decode는 같은 Q8_1 activation을 두 단일 projection launch가
/// 재사용한다. cu206 RTX 3090 Qwen3.6 27B에서 fused 2-accumulator kernel보다
/// register/live-state가 작아 100-token generation이 5.38% 빨랐고 출력은 exact였다.
/// 진단 비교는 `RNB_CUDA_Q4K_GATE_UP_Q8DOT_SPLIT=0`으로 기존 fused kernel을 되켠다.
pub fn q4k_gate_up_q8dot_split_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_GATE_UP_Q8DOT_SPLIT", true)
}

/// Q6_K blocks are 210 bytes, so ql/qh pointers are always 2-byte aligned but
/// alternate between 4-byte aligned and +2. cu207 replaced four byte loads with
/// two halfword loads and cut the 32-token Q6 q8dot kernel sum by 34.97%.
/// `RNB_CUDA_Q6K_Q8DOT_HALF2=0` restores the byte-load diagnostic path.
pub fn q6k_q8dot_half2_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_Q8DOT_HALF2", true)
}

/// Qwen3.6 GDN의 head_k_dim=128 decode는 4-warp reduction으로 두 번의
/// 256-thread shared-memory tree를 대체한다. cu208 RTX 3090에서 delta kernel
/// 합계가 70.514→18.747ms/32 tokens로 줄고 100-token generation은 1.72%
/// 개선됐으며 raw/chat 출력 hash가 각각 exact였다. 진단 대조는
/// `RNB_CUDA_GDN_DELTA_WARP128=0`으로 기존 reduction을 되켠다.
pub fn gdn_delta_warp128_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_DELTA_WARP128", true)
}

/// cu212: 2-token verify window 의 weight-read-once q8dot GEMV. 토큰별 산술
/// 순서가 배치 커널의 해당 seq CTA 와 동일해 per-token bitwise 로 같고, weight
/// bytes 만 절반을 읽는다. `RNB_CUDA_Q{4,5,6}K_Q8DOT_PAIR2=0` 은 per-token CTA
/// 배치 커널로 되돌리는 진단 opt-out 이다.
pub fn q4k_q8dot_pair2_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_Q8DOT_PAIR2", true)
}

pub fn q5k_q8dot_pair2_enabled() -> bool {
    env_bool("RNB_CUDA_Q5K_Q8DOT_PAIR2", true)
}

pub fn q6k_q8dot_pair2_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_Q8DOT_PAIR2", true)
}

/// cu216: Q4_K q8dot family 의 wide-lane 세대(lane 당 8 elem, sc/mn·ds 반감,
/// 64-bit q/x load). lane partial 배치가 바뀌므로 출력 low bits 가 기존 세대와
/// 다르다 — 단일/batch/pair2/gate_up/qkv 5종이 이 게이트 하나로 함께 전환돼
/// 상호 bitwise 계약이 유지된다. RTX 3090 27B에서 MTP verify −11%/round,
/// 100-token 제품 hash 는 두 세대가 동일했다. `RNB_CUDA_Q4K_Q8DOT_WIDE=0`
/// 은 기존 (j0,j1) 세대로 되돌리는 진단 opt-out 이다.
pub fn q4k_q8dot_wide_enabled() -> bool {
    env_bool("RNB_CUDA_Q4K_Q8DOT_WIDE", true)
}

/// cu219: Q5_K q8dot family 의 wide-lane 세대. Q5_K 176-byte 블록은 qh/qs 가
/// 8-byte 정렬이라 Q4 wide 의 uint2 패턴을 그대로 이식했다 (lane 당 8 elem,
/// sc/mn·ds 해석 반감, qh 도 uint2 한 번). lane partial 배치가 바뀌므로 출력
/// low bits 가 기존 (j0,j1) 세대와 다르다 — 단일/batch/pair2 3종이 이 게이트
/// 하나로 함께 전환돼 상호 bitwise 계약이 유지된다.
/// `RNB_CUDA_Q5K_Q8DOT_WIDE=0` 은 기존 세대로 되돌리는 진단 opt-out 이다.
pub fn q5k_q8dot_wide_enabled() -> bool {
    env_bool("RNB_CUDA_Q5K_Q8DOT_WIDE", true)
}

/// cu219: Q6_K q8dot family 의 wide-lane 세대. 210-byte 블록은 2-byte 정렬뿐이라
/// ql/qh 는 cu207 halfword load 를 유지하고, lane 매핑만 8 연속 elem 으로 넓혀
/// sc/ds 해석을 반감하고 x 를 64-bit 로 읽는다. wide 가 켜지면
/// `RNB_CUDA_Q6K_Q8DOT_HALF2` 는 무시된다 (wide 커널은 항상 halfword load).
/// 단일/batch/pair2 3종이 이 게이트 하나로 함께 전환된다.
/// `RNB_CUDA_Q6K_Q8DOT_WIDE=0` 은 기존 half2/byte 세대로 되돌리는 진단 opt-out.
pub fn q6k_q8dot_wide_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_Q8DOT_WIDE", true)
}

pub fn q4k_prefill_f32_gemm_enabled() -> bool {
    expanded_env_bool("RNB_CUDA_Q4K_PREFILL_F32_GEMM", false)
}

pub fn qwen35_shared_q4_f32_cache_enabled_for_seq(seq_len: usize) -> bool {
    expanded_weight_cache_allowed()
        && (q4k_prefill_f32_gemm_enabled()
            || (env_bool("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", false)
                && qwen35_short_window_prefill(seq_len)))
}

pub fn qwen35_full_layer_shared_q4_f32_cache_enabled() -> bool {
    expanded_weight_cache_allowed()
        && (q4k_prefill_f32_gemm_enabled()
            || env_bool("RNB_CUDA_QWEN35_FULL_LAYER_SHARED_Q4_F32_CACHE", false))
}

pub fn q4_f32_release_after_prefill_enabled() -> bool {
    env_bool("RNB_CUDA_Q4_F32_RELEASE_AFTER_PREFILL", false)
}

pub fn q6k_gemv_batch_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_GEMV_BATCH_WARP8", true)
}

pub fn q6k_gemv_batch_seq2_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_GEMV_BATCH_SEQ2_WARP8", true)
}

pub fn q6k_gemv_batch_seq4_warp8_enabled(
    seq_len: usize,
    rows: usize,
    blocks_per_row: usize,
) -> bool {
    let default = seq_len >= 64 && rows >= 64 && blocks_per_row >= 4;
    env_bool("RNB_CUDA_Q6K_GEMV_BATCH_SEQ4_WARP8", default)
}

pub fn resident_q4k_touch_hits_enabled() -> bool {
    env_bool("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS", false)
}

pub fn resident_q4k_arena_enabled() -> bool {
    env_bool("RNB_CUDA_RESIDENT_Q4K_ARENA", false)
}

pub fn glm_direct_file_prefill_enabled(auto_enabled: bool) -> bool {
    let Ok(value) = std::env::var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL") else {
        return auto_enabled;
    };
    let value = value.to_ascii_lowercase();
    !matches!(value.as_str(), "0" | "false" | "off" | "no")
}

pub fn glm_direct_file_pipeline_enabled() -> bool {
    env_bool("RNB_CUDA_GLM_DIRECT_FILE_PIPELINE", true)
}

pub fn glm_direct_file_expert_stream_enabled() -> bool {
    env_bool("RNB_CUDA_GLM_DIRECT_FILE_EXPERT_STREAM", true)
}

pub fn glm_direct_file_io_uring_enabled() -> bool {
    let Ok(value) = std::env::var("RNB_CUDA_GLM_DIRECT_FILE_IO_URING") else {
        return cfg!(target_os = "linux");
    };
    let value = value.to_ascii_lowercase();
    !matches!(value.as_str(), "0" | "false" | "off" | "no")
}

pub fn glm_direct_file_io_uring_forced() -> bool {
    std::env::var_os("RNB_CUDA_GLM_DIRECT_FILE_IO_URING").is_some()
}

pub fn glm_direct_file_io_uring_queue_depth(request_count: usize) -> usize {
    let default = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    std::env::var("RNB_CUDA_GLM_DIRECT_FILE_IO_URING_DEPTH")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
        .min(request_count.max(1))
}

pub fn glm_expert_grouped_enabled(token_count: usize, slot_count: usize) -> bool {
    token_count > 1 && slot_count > token_count && env_bool("RNB_CUDA_GLM_EXPERT_GROUPED", true)
}

pub fn glm_expert_parallel_enabled() -> bool {
    env_bool("RNB_CUDA_GLM_EXPERT_PARALLEL", false)
}

pub fn glm_expert_parallel_secondary_device(primary_ordinal: i32) -> i32 {
    std::env::var("RNB_CUDA_GLM_EXPERT_PARALLEL_SECONDARY_DEVICE")
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .unwrap_or(if primary_ordinal == 0 { 1 } else { 0 })
}

pub fn glm_expert_parallel_primary_slots(slot_count: usize) -> usize {
    if slot_count < 2 {
        return slot_count;
    }
    std::env::var("RNB_CUDA_GLM_EXPERT_PARALLEL_PRIMARY_SLOTS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(slot_count.div_ceil(2))
        .clamp(1, slot_count - 1)
}

pub fn resident_q4k_batch_pinned_staging_enabled(slab_bytes: usize, missing_len: usize) -> bool {
    if let Ok(value) = std::env::var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED") {
        let value = value.to_ascii_lowercase();
        return !matches!(value.as_str(), "0" | "false" | "off" | "no");
    }
    if missing_len < 2 || slab_bytes == 0 {
        return false;
    }
    let min_bytes = std::env::var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(2 * 1024 * 1024);
    slab_bytes >= min_bytes
}

pub fn qwen35_decode_resident_batch_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DECODE_RESIDENT_BATCH", false)
}

pub fn qwen35_prefill_hot_resident_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT", false)
}

pub fn qwen35_prefill_hot_resident_min_tokens() -> usize {
    std::env::var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MIN_TOKENS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(128)
}

pub fn qwen35_prefill_hot_resident_budget_bytes(resident_q4k_limit: usize) -> usize {
    std::env::var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MB")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .map(|mib| mib.saturating_mul(1024 * 1024))
        .unwrap_or_else(|| (resident_q4k_limit / 512).clamp(8 * 1024 * 1024, 16 * 1024 * 1024))
}

pub fn mtp_expert_trace_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_EXPERT_TRACE", false)
}

pub fn mtp_expert_hot_resident_enabled() -> bool {
    match std::env::var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => env_bool("RNB_MTP_DEVICE_VERIFY", false),
    }
}

pub fn cubin_modules_enabled() -> bool {
    env_bool("RNB_CUDA_CUBIN_MODULES", true)
}

pub fn mtp_verify_output_q6k_token2_enabled(token_count: usize) -> bool {
    token_count == 2 && env_bool("RNB_CUDA_MTP_VERIFY_OUTPUT_Q6K_TOKEN2", true)
}

pub fn q8_0_gemv_batch_token2_enabled(seq_len: usize) -> bool {
    seq_len == 2 && env_bool("RNB_CUDA_Q8_0_GEMV_BATCH_TOKEN2", true)
}

pub fn mtp_verify_gdn_qkv_warp_enabled(window_tokens: usize) -> bool {
    window_tokens == 1
        || (window_tokens == 2 && env_bool("RNB_CUDA_MTP_VERIFY_GDN_QKV_WARP2", true))
}

pub fn mtp_verify_router_stable_key_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_ROUTER_STABLE_KEY", true)
}

/// cu219: dense verify GDN 경로의 model F32 weight lookup(alpha/beta, conv
/// kernel, dt_bias/ssm_a, ssm_norm)을 content FNV hash 대신 allocation-identity
/// 키로 조회한다. 이 slice 들은 target decode chain 이 이미 stable 키로 쓰는
/// 것과 같은 model tensor 라 engine 수명 동안 주소·내용이 안정적이고, engine
/// reset 은 `clear_stable_resident_f32_sources` 로 stable 항목을 함께 비운다.
/// 27B verify 는 layer 당 alpha/beta 해싱에만 host ~163µs 를 쓰고 있었다
/// (nsys idle 421.8ms/run — Q4 pair2 → f32 multi2 경계).
/// `RNB_CUDA_MTP_VERIFY_GDN_STABLE_KEYS=0` 은 content-hash 조회로 되돌리는
/// 진단 opt-out 이다.
pub fn mtp_verify_gdn_stable_keys_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_GDN_STABLE_KEYS", true)
}

pub fn mtp_verify_snapshot_pool_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SNAPSHOT_POOL", true)
}

pub fn mtp_expert_extra_resident_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT", true)
}

pub fn mtp_expert_extra_resident_budget_bytes(resident_q4k_limit: usize) -> usize {
    mtp_expert_extra_resident_budget_bytes_for_layer(resident_q4k_limit, 0)
}

pub fn mtp_expert_extra_resident_budget_bytes_for_layer(
    resident_q4k_limit: usize,
    layer_observations: usize,
) -> usize {
    if !mtp_expert_extra_resident_enabled() {
        return 0;
    }
    let cold_bytes = resident_q4k_limit / 256;
    let warm_bytes = (resident_q4k_limit / 128)
        .min(8 * 1024 * 1024)
        .max(cold_bytes);
    let default_bytes = if layer_observations >= 8 {
        warm_bytes
    } else {
        cold_bytes
    };
    let Some(raw) = std::env::var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB").ok() else {
        return default_bytes;
    };
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return default_bytes;
    }
    raw.parse::<usize>()
        .map(|mib| mib.saturating_mul(1024 * 1024))
        .unwrap_or(default_bytes)
}

pub fn qwen35_decode_q4k_arena_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA", true)
}

pub fn prefill_output_logits_requested() -> bool {
    env_bool("RNB_CUDA_PREFILL_OUTPUT_LOGITS", true)
}

pub fn prefill_gemv_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_GEMV", true)
}

// cu29 Phase 2: hd=128 fused QKV+RoPE+f16-pack path. Llama / Mistral 등
// qk-norm 없는 dense hd128 모델에서 host RoPE round-trip 제거. nsys 진단으로
// D2H sync wait 81% 가 진짜 lever 확정 (cu28). 측정 후 default ON 가능.
pub fn hd128_fused_qkv_rope_enabled() -> bool {
    env_bool("RNB_CUDA_HD128_FUSED_QKV_ROPE", false)
}

pub fn prefill_q4k_f16_gemm_enabled() -> bool {
    expanded_env_bool("RNB_CUDA_Q4K_PREFILL_F16_GEMM", false)
}

pub fn prefill_q4k_f16_qkv_gemm_enabled() -> bool {
    expanded_env_bool(
        "RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM",
        prefill_q4k_f16_gemm_enabled(),
    )
}

pub fn prefill_q4k_f16_o_proj_enabled() -> bool {
    expanded_env_bool(
        "RNB_CUDA_Q4K_PREFILL_F16_O_PROJ",
        prefill_q4k_f16_gemm_enabled(),
    )
}

pub fn prefill_q4k_f16_o_proj_force_enabled() -> bool {
    expanded_env_force("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ")
}

pub fn prefill_delta_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_DELTA", true)
}

pub fn prefill_delta_k128_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_DELTA_K128_WARP4", true)
}

pub fn prefill_moe_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_MOE", true)
}

pub fn prefill_moe_enabled_for_seq(seq_len: usize) -> bool {
    prefill_moe_enabled() && !qwen35_short_window_prefill(seq_len)
}

pub fn prefill_moe_full_layer_enabled() -> bool {
    // 2026-05-26: Qwen3.6 35B full-layer prefill probe hit CUDA 719 and
    // kernel Xid79 on RTX 3080. Keep the original env quarantined unless the
    // caller also opts into the explicit retry gate and the device-side slot
    // pointer path that survived the controlled RTX 3080 retry.
    env_bool("RNB_CUDA_PREFILL_MOE_FULL_LAYER", false)
        && env_bool("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY", false)
        && qwen35_full_layer_device_slot_ptrs_enabled()
}

pub fn prefill_moe_full_layer_min_expert_permille() -> usize {
    std::env::var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(750)
        .clamp(1, 1000)
}

pub fn prefill_moe_weight_prefetch_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH", false)
}

pub fn prefill_moe_weight_prefetch_pinned_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH_PINNED", false)
}

pub fn prefill_moe_range_slab_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_MOE_RANGE_SLAB", false)
}

pub fn prefill_moe_range_slab_max_gap_experts() -> usize {
    std::env::var("RNB_CUDA_PREFILL_MOE_RANGE_SLAB_MAX_GAP_EXPERTS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(3)
}

pub fn prefill_moe_range_slab_max_overhead_permille() -> usize {
    std::env::var("RNB_CUDA_PREFILL_MOE_RANGE_SLAB_MAX_OVERHEAD_PERMILLE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(1250)
        .max(1000)
}

pub fn qwen35_full_layer_device_slot_ptrs_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS", false)
}

pub fn moe_layer_cache_enabled() -> bool {
    env_bool("RNB_CUDA_MOE_LAYER_CACHE", false)
}
pub fn mtp_verify_resident_moe_layer_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_RESIDENT_MOE_LAYER", true)
}

pub fn mtp_verify_missing_moe_layer_promotion_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_MISSING_MOE_LAYER_PROMOTION", false)
}

pub fn mtp_verify_resident_conv_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_RESIDENT_CONV", true)
}

pub fn mtp_verify_resident_attn_kv_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_RESIDENT_ATTN_KV", true)
}

pub fn mtp_verify_gdn_graph_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_GDN_GRAPH", true)
}

pub fn mtp_verify_window2_graphs_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_WINDOW2_GRAPHS", true)
}

pub fn mtp_verify_attention_graph_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_ATTENTION_GRAPH", true)
}

pub fn mtp_verify_q8_multi_projection_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_Q8_MULTI_PROJECTION", true)
}

pub fn mtp_verify_f32_multi_projection_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_F32_MULTI_PROJECTION", true)
}

pub fn mtp_verify_shared_scale_add_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SHARED_SCALE_ADD", true)
}

pub fn mtp_verify_segment_graph_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SEGMENT_GRAPH", true)
}
pub fn gemma_mtp2_finalize_megakernel_enabled() -> bool {
    env_bool("RNB_CUDA_GEMMA_MTP2_FINALIZE_M1", true)
}

pub fn mtp_verify_selected_q8_gate_up_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_Q8_GATE_UP", true)
}

pub fn mtp_verify_selected_gate_pair2_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2", true)
}

pub fn mtp_verify_selected_gate_pair2_silu_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2_SILU", true)
}

pub fn mtp_verify_selected_down_pair2_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_PAIR2", true)
}

pub fn mtp_verify_selected_pair_map_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_PAIR_MAP", true)
}

pub fn mtp_verify_selected_gate_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8", true)
}

pub fn mtp_verify_selected_gate_warp_reduce_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP_REDUCE", true)
}

pub fn mtp_verify_selected_down_warp8_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_WARP8", true)
}

pub fn q6k_argmax_batched_single_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_ARGMAX_BATCHED_SINGLE", true)
}

pub fn q6k_warp8_unrolled_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_WARP8_UNROLLED", true)
}

pub fn prefill_f32_gemm_allowed(
    quant_supported: bool,
    seq_len: usize,
    rows: usize,
    cols: usize,
) -> bool {
    if !env_bool("RNB_CUDA_PREFILL_F32_GEMM", true) || seq_len <= 1 || !quant_supported {
        return false;
    }
    let min_seq = std::env::var("RNB_CUDA_PREFILL_F32_GEMM_MIN_SEQ")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(128);
    if seq_len < min_seq {
        return false;
    }
    let max_rows = std::env::var("RNB_CUDA_PREFILL_F32_GEMM_MAX_ROWS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(8192);
    let max_cols = std::env::var("RNB_CUDA_PREFILL_F32_GEMM_MAX_COLS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(4096);
    rows <= max_rows && cols <= max_cols
}

pub fn prefill_f32_gemm_trace_enabled() -> bool {
    env_is_one("RNB_CUDA_F32_GEMM_TRACE")
}

pub fn layer_gemv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_CUDA_LAYER_GEMV", true))
}

pub fn delta_net_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("RNB_CUDA_DELTA_NET") {
        Ok(_) => env_bool("RNB_CUDA_DELTA_NET", true),
        Err(_) => layer_gemv_enabled(),
    })
}

pub fn delta_state_sync_each_step_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_CUDA_DELTA_STATE_SYNC_EACH_STEP", false))
}

pub fn decode_attention_enabled() -> bool {
    env_is_one("RNB_CUDA_DECODE_ATTN")
}

pub fn decode_attention_kv_cache_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_KV_CACHE", true)
}

pub fn decode_attention_sliding_window_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_SLIDING_WINDOW", false)
}

pub fn decode_attention_hd512_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_HD512", true)
}

pub fn decode_attention_hd256_split_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_HD256_SPLIT", true)
}

pub fn decode_attention_hd256_split_chunk_size() -> usize {
    std::env::var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&chunk| matches!(chunk, 128 | 256 | 512 | 1024))
        .unwrap_or(256)
}

pub fn mtp_verify_attention_hd256_split_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_ATTN_HD256_SPLIT", true)
}

pub fn mtp_verify_attention_hd256_query_tile_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_ATTN_HD256_QUERY_TILE", true)
}

fn cuda_arch_supports_ampere_mma(arch: &str) -> bool {
    arch.strip_prefix("sm_")
        .and_then(|cc| cc.parse::<u32>().ok())
        .is_some_and(|cc| cc >= 80)
}

pub fn compiled_ampere_mma_supported() -> bool {
    option_env!("RNB_CUDA_COMPILED_ARCH").is_some_and(cuda_arch_supports_ampere_mma)
}

pub fn mtp_verify_attention_hd256_mma_stream_k_enabled() -> bool {
    compiled_ampere_mma_supported() && env_bool("RNB_CUDA_MTP_ATTN_HD256_MMA_STREAM_K", true)
}

pub fn mtp_verify_attention_hd256_split_chunk_size() -> usize {
    std::env::var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&chunk| matches!(chunk, 128 | 256 | 512 | 1024))
        .unwrap_or_else(decode_attention_hd256_split_chunk_size)
}

pub fn decode_attention_hd512_split_enabled() -> bool {
    env_bool("RNB_CUDA_DECODE_ATTN_HD512_SPLIT", true)
}

pub fn decode_attention_hd512_split_chunk_size() -> usize {
    std::env::var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&chunk| matches!(chunk, 128 | 256 | 512 | 1024))
        .unwrap_or(512)
}

pub fn gdn_prefill_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL", false)
}

pub fn gdn_prefill_chain_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN", true)
}

pub fn gdn_prefill_chain_device_output_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_DEVICE_OUTPUT", false)
}

pub fn gdn_prefill_chain_moe_input_device_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_INPUT_DEVICE", true)
}

pub fn gdn_prefill_chain_moe_output_device_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_OUTPUT_DEVICE", true)
}

pub fn gdn_prefill_chain_dense_ffn_device_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_DENSE_FFN_DEVICE", true)
}

pub fn gdn_prefill_chain_skip_host_projection_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_PREFILL_CHAIN_SKIP_HOST_PROJECTION", true)
}

pub fn qwen35_device_moe_phase_profile_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE", false)
}

pub fn qwen35_device_moe_inplace_residual_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL", true)
}

pub fn qwen35_q4_gate_up_silu_fused_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED", true)
}

pub fn qwen35_q4_gate_up_q8dot_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", true)
}

pub fn qwen35_q4_gate_up_q8dot_q4_down_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_Q4_DOWN", true)
}

pub fn qwen35_q4_down_q8dot_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT", true)
}
pub fn qwen35_q4_gate_up_q8dot_mmq_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ", true)
}

pub fn qwen35_q5_down_q8dot_mmq_enabled(token_count: usize) -> bool {
    token_count >= 32 && env_bool("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ", true)
}

pub fn qwen35_q5_down_q8dot_mmq_group32_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP32", true)
}

pub fn qwen35_q5_down_q8dot_mmq_group64_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP64", true)
}

pub fn qwen35_q4_gate_up_q8dot_mmq_group16_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16", true)
}

pub fn qwen35_q4_gate_up_q8dot_mmq_group32_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", true)
}

pub fn qwen35_q4_gate_up_q8_handoff_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF", true)
}

pub fn qwen35_selected_base_stream_enabled(token_count: usize) -> bool {
    match std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_STREAM") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => token_count >= 32,
    }
}

pub fn qwen35_q4_gate_up_silu_pack4_f32_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_PACK4_F32", true)
}

pub fn prefill_conv_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_CONV", true)
}

pub fn prefill_temp_coalesce_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_TEMP_COALESCE", false)
}

pub fn prefill_temp_run_coalesce_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE", true)
}

pub fn prefill_temp_pinned_staging_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_TEMP_PINNED_STAGING", false)
}

pub fn prefill_temp_host_register_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER", false)
}

pub fn prefill_temp_host_register_min_slots() -> usize {
    std::env::var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_SLOTS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(128)
}

pub fn prefill_temp_host_register_granularity_bytes() -> usize {
    const DEFAULT_GRANULARITY_BYTES: usize = 4096;
    const MIN_GRANULARITY_BYTES: usize = 4096;
    const MAX_GRANULARITY_BYTES: usize = 64 * 1024 * 1024;
    let bytes = std::env::var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_GRANULARITY_KB")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .and_then(|kb| kb.checked_mul(1024))
        .unwrap_or(DEFAULT_GRANULARITY_BYTES)
        .clamp(MIN_GRANULARITY_BYTES, MAX_GRANULARITY_BYTES);
    bytes.next_power_of_two()
}

pub fn prefill_temp_host_register_min_bytes() -> usize {
    std::env::var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_BYTES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(1024 * 1024)
}

/// Fuse the DeepSeek4 attention output projection into one device-resident
/// sequence. Diagnostic opt-out for A/B against the per-group host GEMV loop.
pub fn deepseek4_output_projection_fused_enabled() -> bool {
    env_bool("RNB_CUDA_DEEPSEEK4_OUTPUT_FUSED", true)
}

/// Serve model-owned F32 GEMM weights from the resident cache instead of
/// re-uploading them per call. Diagnostic opt-out.
pub fn f32_gemm_resident_weights_enabled() -> bool {
    env_bool("RNB_CUDA_F32_GEMM_RESIDENT_WEIGHTS", true)
}

pub fn prefill_down_copy_overlap_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP", false)
}

pub fn prefill_moe_sync_before_sparse_enabled() -> bool {
    // The sparse phase can grow shared scratch buffers. Allowing this sync to
    // be disabled risks freeing queued shared-phase buffers before the stream
    // has consumed them.
    let _ = std::env::var("RNB_CUDA_PREFILL_MOE_SYNC_BEFORE_SPARSE");
    true
}

pub fn group4_down_row8_enabled() -> bool {
    env_bool("RNB_CUDA_GROUP4_DOWN_ROW8", false)
}

pub fn qwen35_q4_down_group4_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", true)
}

pub fn qwen35_down_token_major_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_DOWN_TOKEN_MAJOR", false)
}

pub fn qwen35_q6_down_full4_split_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_SPLIT", false)
}

pub fn qwen35_q6_down_full4_fastpath_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_FULL4_FASTPATH", false)
}

pub fn qwen35_q6_down_q8dot_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_Q8DOT", false)
}

pub fn qwen35_q6_down_run_batched_ref_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED_REF", false)
}

pub fn qwen35_q6_down_run_batched8_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_RUN_BATCHED8", false)
}

pub fn qwen35_q6_down_run_tiled4_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_RUN_TILED4", false)
}

pub fn qwen35_q6_down_pack4_f32_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32", true)
}

pub fn qwen35_q6_down_pack4_f32_vec4_enabled() -> bool {
    env_bool("RNB_CUDA_QWEN35_Q6_DOWN_PACK4_F32_VEC4", true)
}

pub fn group2_down_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_GROUP2_DOWN_WARP4", false)
}

pub fn mtp_verify_group2_down_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_MTP_VERIFY_GROUP2_DOWN_WARP4", false)
}

pub fn q6k_group4_down_lowreg_enabled() -> bool {
    env_bool("RNB_CUDA_Q6K_GROUP4_DOWN_LOWREG", false)
}

pub fn gdn_gated_norm_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_GATED_NORM", true)
}

pub fn gdn_gated_norm_gemm_enabled() -> bool {
    env_bool("RNB_CUDA_GDN_GATED_NORM_GEMM", true)
}

pub fn gdn_gated_norm_gemm_enabled_for_seq(seq_len: usize) -> bool {
    gdn_gated_norm_gemm_enabled() && !qwen35_short_window_prefill(seq_len)
}

/// cu219: GDN prefill projection 모드. 기본 `auto`는 quant가 batch GEMV를
/// 지원하면 원본 quant weight를 그대로 쓰고(업로드 0 — resident 재사용),
/// 미지원 quant만 host dequant F32 GEMM으로 후퇴한다. 이전 기본 `f32`는
/// seq>2에서 층당 dequant F32 전량을 미등록 직접 업로드해(27B: 층당
/// 440MiB, 요청당 ~20GiB, prefill의 ~2.2s) cu194 클래스의 매 요청 재업로드를
/// 만들었다. 명시 `f32`/`q`는 그대로 존중한다.
pub fn gdn_prefill_gemv_mode() -> Option<String> {
    let raw = std::env::var("RNB_CUDA_GDN_PREFILL_GEMV").unwrap_or_else(|_| "auto".to_string());
    let mode = raw.to_ascii_lowercase();
    if matches!(mode.as_str(), "0" | "false" | "off" | "no") {
        None
    } else if matches!(mode.as_str(), "" | "1" | "true" | "on" | "yes") {
        Some("auto".to_string())
    } else {
        Some(mode)
    }
}

pub fn gdn_prefill_gemv_mode_for_seq(seq_len: usize) -> Option<String> {
    if qwen35_short_window_prefill(seq_len) {
        None
    } else {
        gdn_prefill_gemv_mode()
    }
}

pub fn qwen35_short_window_prefill(seq_len: usize) -> bool {
    seq_len > 1 && seq_len <= qwen35_short_window_prefill_max_seq()
}

fn qwen35_short_window_prefill_max_seq() -> usize {
    std::env::var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0)
}

pub fn prefill_flash_attention_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_FLASH_ATTN", true)
}

pub fn prefill_flash_attention_hd512_w256_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256", true)
}

pub fn prefill_flash_attention_min_seq(head_dim: usize) -> usize {
    let env_name = match head_dim {
        128 => "RNB_CUDA_PREFILL_FLASH_ATTN_HD128_MIN_SEQ",
        256 => "RNB_CUDA_PREFILL_FLASH_ATTN_HD256_MIN_SEQ",
        512 => "RNB_CUDA_PREFILL_FLASH_ATTN_HD512_MIN_SEQ",
        _ => "RNB_CUDA_PREFILL_FLASH_ATTN_MIN_SEQ",
    };
    std::env::var(env_name)
        .or_else(|_| std::env::var("RNB_CUDA_PREFILL_FLASH_ATTN_MIN_SEQ"))
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(match head_dim {
            128 => 128,
            _ => 1,
        })
}

pub fn moe_route_hist_enabled() -> bool {
    env_is_one("RNB_CUDA_MOE_ROUTE_HIST")
}

pub fn shared_f32_enabled() -> bool {
    std::env::var("RNB_CUDA_SHARED_F32").ok().as_deref() != Some("0")
}

pub fn nemotron_q5_full_layer_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q5_FULL_LAYER", false)
}

pub fn nemotron_q5_layer_cache_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q5_LAYER_CACHE", true)
}

pub fn nemotron_q8_shared_q5_sparse_decode_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q8_SHARED_Q5_SPARSE_DECODE", true)
}

pub fn nemotron_q8_shared_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q8_SHARED_WARP4", true)
}

pub fn nemotron_q8_shared_cublas_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q8_SHARED_CUBLAS", true)
}

pub fn nemotron_decode_sparse_copy_prefetch_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_DECODE_COPY_PREFETCH", false)
}

pub fn nemotron_prefill_sparse_copy_prefetch_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_PREFILL_COPY_PREFETCH", false)
}

pub fn nemotron_prefill_sparse_input_pinned_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED", false)
}

pub fn nemotron_prefill_q8_shared_fused_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_FUSED", false)
}

pub fn nemotron_prefill_q8_shared_sparse_fused_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_PREFILL_Q8_SHARED_SPARSE_FUSED", false)
}

pub fn nemotron_prefill_group4_enabled(token_count: usize, slots: usize) -> bool {
    if !env_bool("RNB_CUDA_NEMOTRON_PREFILL_GROUP4", true) {
        return false;
    }
    let min_tokens = std::env::var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(16);
    let min_slots = std::env::var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(64);
    token_count >= min_tokens && slots >= min_slots
}

pub fn prefill_q8_0_batch_enabled() -> bool {
    env_bool("RNB_CUDA_PREFILL_Q8_0_BATCH", false)
}

pub fn nemotron_q5_down_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q5_DOWN_WARP4", false)
}

pub fn nemotron_q8_down_warp4_enabled() -> bool {
    env_bool("RNB_CUDA_NEMOTRON_Q8_DOWN_WARP4", false)
}

pub fn qwen_moe_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_QWEN_MOE_CUDA_GATE_UP", true))
}

pub fn qwen_moe_batch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_CUDA_BATCH_MOE", qwen_moe_gate_up_enabled()))
}

pub fn qwen_moe_device_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_bool("RNB_CUDA_DEVICE_DECODE", true))
}

pub fn cu65_device_qkv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_one("RNB_CU65_DEVICE_QKV"))
}

pub fn cu68_layer_graph_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_one("RNB_CU68_LAYER_GRAPH"))
}

pub fn cu63_device_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_one("RNB_CU63_DEVICE_DECODE"))
}

pub fn cu63_sync_diag() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_one("RNB_CU63_SYNC_DIAG"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cuda_runtime_policy() {
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_OUTPUT_LOGITS");
            std::env::remove_var("RNB_CUDA_OUTPUT_ARGMAX");
            std::env::remove_var("RNB_CUDA_GDN_PREFILL");
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN");
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_CHAIN_MOE_INPUT_DEVICE");
            std::env::remove_var("RNB_CUDA_PREFILL_CONV");
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_COALESCE");
            std::env::remove_var("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP");
            std::env::remove_var("RNB_CUDA_PREFILL_DELTA_K128_WARP4");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_PINNED_STAGING");
            std::env::remove_var("RNB_CUDA_GROUP4_DOWN_ROW8");
            std::env::remove_var("RNB_CUDA_GROUP2_DOWN_WARP4");
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_GROUP2_DOWN_WARP4");
            std::env::remove_var("RNB_CUDA_Q6K_GROUP4_DOWN_LOWREG");
            std::env::remove_var("RNB_CUDA_Q6K_OUTPUT_WARP8");
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP8");
            std::env::remove_var("RNB_CUDA_Q4K_GEMV_WARP8");
            std::env::remove_var("RNB_CUDA_Q4K_PACKED_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_Q6K_PACKED_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_MOE_GRAPH");
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH");
            std::env::remove_var("RNB_CUDA_Q4K_GEMV_BATCH_WARP8");
            std::env::remove_var("RNB_CUDA_Q6K_GEMV_BATCH_WARP8");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_ARENA");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES");
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE");
            std::env::remove_var("RNB_CUDA_MOE_LAYER_CACHE");
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_RESIDENT_MOE_LAYER");
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_MISSING_MOE_LAYER_PROMOTION");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_TRACE");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB");
            std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
            std::env::remove_var("RNB_CUDA_PREFILL_OUTPUT_LOGITS");
            std::env::remove_var("RNB_CUDA_DELTA_STATE_SYNC_EACH_STEP");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS");
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ");
            std::env::remove_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4");
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
            std::env::remove_var("RNB_CUDA_Q4K_GATE_UP_Q8DOT_SPLIT");
            std::env::remove_var("RNB_CUDA_Q6K_Q8DOT_HALF2");
            std::env::remove_var("RNB_CUDA_GDN_DELTA_WARP128");
            std::env::remove_var("RNB_CUDA_Q4K_Q8DOT_PAIR2");
            std::env::remove_var("RNB_CUDA_Q5K_Q8DOT_PAIR2");
            std::env::remove_var("RNB_CUDA_Q6K_Q8DOT_PAIR2");
            std::env::remove_var("RNB_CUDA_Q4K_Q8DOT_WIDE");
            std::env::remove_var("RNB_CUDA_Q5K_Q8DOT_WIDE");
            std::env::remove_var("RNB_CUDA_Q6K_Q8DOT_WIDE");
        }
        assert!(output_logits_enabled());
        assert!(prefill_output_logits_requested());
        assert!(!output_argmax_enabled());
        assert!(q6k_output_warp8_enabled());
        assert!(q8_0_gemv_warp4_enabled());
        assert!(q8_0_gemv_warp8_enabled());
        assert!(q4k_gemv_warp8_enabled());
        assert!(!q4k_packed_gemv_warp4_enabled());
        assert!(!q6k_packed_gemv_warp4_enabled());
        assert!(qwen35_decode_moe_graph_enabled());
        assert!(qwen35_selected_sparse_compound_graph_enabled());
        assert!(!q8_0_output_q8dot_argmax_enabled());
        assert!(q4k_gemv_batch_warp8_enabled());
        assert!(q4k_batch_raw_seq4_enabled(1115, 2560, 10));
        assert!(!q4k_batch_raw_seq4_enabled(4, 2560, 10));
        assert!(q6k_packed_batch_warp4_enabled(14));
        assert!(!q6k_packed_batch_warp4_enabled(7));
        assert!(q6k_gemv_batch_warp8_enabled());
        assert!(q4k_gate_up_q8dot_split_enabled());
        assert!(q6k_q8dot_half2_enabled());
        assert!(gdn_delta_warp128_enabled());
        assert!(q4k_q8dot_pair2_enabled());
        assert!(q5k_q8dot_pair2_enabled());
        assert!(q6k_q8dot_pair2_enabled());
        assert!(q4k_q8dot_wide_enabled());
        assert!(q5k_q8dot_wide_enabled());
        assert!(q6k_q8dot_wide_enabled());
        assert!(!resident_q4k_touch_hits_enabled());
        assert!(!resident_q4k_arena_enabled());
        assert!(!resident_q4k_batch_pinned_staging_enabled(1024 * 1024, 2));
        assert!(resident_q4k_batch_pinned_staging_enabled(
            2 * 1024 * 1024,
            2
        ));
        assert!(qwen35_decode_q4k_arena_enabled());
        assert!(!gdn_prefill_enabled());
        assert!(gdn_prefill_chain_enabled());
        assert!(gdn_prefill_chain_moe_input_device_enabled());
        assert!(gdn_prefill_chain_moe_output_device_enabled());
        assert!(gdn_prefill_chain_skip_host_projection_enabled());
        assert!(prefill_conv_enabled());
        assert!(!prefill_moe_full_layer_enabled());
        assert!(!moe_layer_cache_enabled());
        assert!(mtp_verify_resident_moe_layer_enabled());
        assert!(!mtp_verify_missing_moe_layer_promotion_enabled());
        assert!(!mtp_expert_trace_enabled());
        assert!(!mtp_expert_hot_resident_enabled());
        assert!(mtp_expert_extra_resident_enabled());
        assert!(!prefill_temp_coalesce_enabled());
        assert!(prefill_temp_run_coalesce_enabled());
        assert!(!prefill_temp_pinned_staging_enabled());
        assert!(!prefill_q4k_f16_gemm_enabled());
        assert!(!prefill_q4k_f16_qkv_gemm_enabled());
        assert!(!prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_down_copy_overlap_enabled());
        assert!(prefill_delta_k128_warp4_enabled());
        assert!(!delta_state_sync_each_step_enabled());
        assert!(!group4_down_row8_enabled());
        assert!(!group2_down_warp4_enabled());
        assert!(!mtp_verify_group2_down_warp4_enabled());
        assert!(!q6k_group4_down_lowreg_enabled());
        assert!(!nemotron_prefill_group4_enabled(15, 64));
        assert!(!nemotron_prefill_group4_enabled(16, 63));
        assert!(nemotron_prefill_group4_enabled(16, 64));
    }

    #[test]
    fn nemotron_prefill_group4_policy_is_model_scoped() {
        unsafe {
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS");
        }
        assert!(!nemotron_prefill_group4_enabled(1, 6));
        assert!(!nemotron_prefill_group4_enabled(16, 32));
        assert!(nemotron_prefill_group4_enabled(16, 64));

        unsafe {
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4", "0");
        }
        assert!(!nemotron_prefill_group4_enabled(128, 768));

        unsafe {
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4", "1");
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS", "32");
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS", "128");
        }
        assert!(!nemotron_prefill_group4_enabled(16, 768));
        assert!(!nemotron_prefill_group4_enabled(128, 64));
        assert!(nemotron_prefill_group4_enabled(128, 768));

        unsafe {
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_TOKENS");
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_GROUP4_MIN_SLOTS");
        }
    }

    #[test]
    fn nemotron_sparse_input_pinned_staging_is_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED");
        }
        assert!(!nemotron_prefill_sparse_input_pinned_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED", "1");
        }
        assert!(nemotron_prefill_sparse_input_pinned_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED", "0");
        }
        assert!(!nemotron_prefill_sparse_input_pinned_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_INPUT_PINNED");
        }
    }

    #[test]
    fn qwen35_device_moe_phase_profile_is_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE");
        }
        assert!(!qwen35_device_moe_phase_profile_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE", "1");
        }
        assert!(qwen35_device_moe_phase_profile_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE", "0");
        }
        assert!(!qwen35_device_moe_phase_profile_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_MOE_PHASE_PROFILE");
        }
    }

    #[test]
    fn qwen35_device_moe_inplace_residual_is_default_on_with_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL");
        }
        assert!(qwen35_device_moe_inplace_residual_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL", "0");
        }
        assert!(!qwen35_device_moe_inplace_residual_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL", "1");
        }
        assert!(qwen35_device_moe_inplace_residual_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DEVICE_MOE_INPLACE_RESIDUAL");
        }
    }

    #[test]
    fn prefill_f32_gemm_applies_shape_thresholds() {
        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM", "1");
            std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM_MIN_SEQ", "8");
            std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_ROWS", "128");
            std::env::set_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_COLS", "256");
        }
        assert!(prefill_f32_gemm_allowed(true, 8, 128, 256));
        assert!(!prefill_f32_gemm_allowed(true, 7, 128, 256));
        assert!(!prefill_f32_gemm_allowed(false, 8, 128, 256));
        assert!(!prefill_f32_gemm_allowed(true, 8, 129, 256));
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_F32_GEMM");
            std::env::remove_var("RNB_CUDA_PREFILL_F32_GEMM_MIN_SEQ");
            std::env::remove_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_ROWS");
            std::env::remove_var("RNB_CUDA_PREFILL_F32_GEMM_MAX_COLS");
        }
    }

    #[test]
    fn gdn_prefill_gemv_defaults_to_f32_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_GEMV");
        }
        assert_eq!(gdn_prefill_gemv_mode().as_deref(), Some("auto"));

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_GEMV", "0");
        }
        assert_eq!(gdn_prefill_gemv_mode(), None);

        unsafe {
            std::env::set_var("RNB_CUDA_GDN_PREFILL_GEMV", "q");
        }
        assert_eq!(gdn_prefill_gemv_mode().as_deref(), Some("q"));

        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_GEMV");
        }
    }

    #[test]
    fn qwen35_short_window_prefill_policy_is_opt_in() {
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_GDN_PREFILL_GEMV");
            std::env::remove_var("RNB_CUDA_GDN_GATED_NORM_GEMM");
            std::env::remove_var("RNB_CUDA_PREFILL_MOE");
            std::env::remove_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ");
            std::env::remove_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM");
            std::env::remove_var("RNB_CUDA_Q4_F32_RELEASE_AFTER_PREFILL");
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
        }

        assert_eq!(gdn_prefill_gemv_mode_for_seq(1).as_deref(), Some("auto"));
        assert!(gdn_gated_norm_gemm_enabled_for_seq(1));
        assert!(prefill_moe_enabled_for_seq(1));
        assert!(!qwen35_shared_q4_f32_cache_enabled_for_seq(1));

        assert_eq!(gdn_prefill_gemv_mode_for_seq(2).as_deref(), Some("auto"));
        assert!(gdn_gated_norm_gemm_enabled_for_seq(2));
        assert!(prefill_moe_enabled_for_seq(2));
        assert!(!qwen35_shared_q4_f32_cache_enabled_for_seq(2));

        unsafe {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
            std::env::set_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE", "1");
            std::env::set_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ", "16");
        }

        assert_eq!(gdn_prefill_gemv_mode_for_seq(16), None);
        assert!(!gdn_gated_norm_gemm_enabled_for_seq(16));
        assert!(!prefill_moe_enabled_for_seq(16));
        assert!(qwen35_shared_q4_f32_cache_enabled_for_seq(16));

        assert_eq!(gdn_prefill_gemv_mode_for_seq(17).as_deref(), Some("auto"));
        assert!(gdn_gated_norm_gemm_enabled_for_seq(17));
        assert!(prefill_moe_enabled_for_seq(17));
        assert!(!qwen35_shared_q4_f32_cache_enabled_for_seq(17));
        assert!(!q4_f32_release_after_prefill_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "1");
        }
        assert!(qwen35_shared_q4_f32_cache_enabled_for_seq(17));
        assert!(!q4_f32_release_after_prefill_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4_F32_RELEASE_AFTER_PREFILL", "1");
        }
        assert!(q4_f32_release_after_prefill_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::remove_var("RNB_CUDA_QWEN35_SHARED_Q4_F32_CACHE");
            std::env::remove_var("RNB_CUDA_QWEN35_SHORT_WINDOW_PREFILL_MAX_SEQ");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM");
            std::env::remove_var("RNB_CUDA_Q4_F32_RELEASE_AFTER_PREFILL");
        }
    }

    #[test]
    fn prefill_moe_full_layer_threshold_is_configurable_permille() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE");
        }
        assert_eq!(prefill_moe_full_layer_min_expert_permille(), 750);

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE", "950");
        }
        assert_eq!(prefill_moe_full_layer_min_expert_permille(), 950);

        unsafe {
            std::env::set_var(
                "RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE",
                "5000",
            );
        }
        assert_eq!(prefill_moe_full_layer_min_expert_permille(), 1000);

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_MIN_EXPERT_PERMILLE");
        }
    }

    #[test]
    fn prefill_moe_weight_prefetch_pinned_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH_PINNED");
        }
        assert!(!prefill_moe_weight_prefetch_pinned_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH_PINNED", "1");
        }
        assert!(prefill_moe_weight_prefetch_pinned_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_WEIGHT_PREFETCH_PINNED");
        }
    }

    #[test]
    fn prefill_temp_coalesce_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_COALESCE");
        }
        assert!(!prefill_temp_coalesce_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_COALESCE", "1");
        }
        assert!(prefill_temp_coalesce_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_COALESCE");
        }
    }

    #[test]
    fn prefill_temp_run_coalesce_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE");
        }
        assert!(prefill_temp_run_coalesce_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE", "0");
        }
        assert!(!prefill_temp_run_coalesce_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_RUN_COALESCE");
        }
    }

    #[test]
    fn glm_direct_file_prefill_uses_constrained_default_and_allows_override() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL");
        }
        assert!(!glm_direct_file_prefill_enabled(false));
        assert!(glm_direct_file_prefill_enabled(true));

        unsafe {
            std::env::set_var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL", "0");
        }
        assert!(!glm_direct_file_prefill_enabled(true));

        unsafe {
            std::env::set_var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL", "1");
        }
        assert!(glm_direct_file_prefill_enabled(false));

        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_DIRECT_FILE_PREFILL");
        }
    }

    #[test]
    fn glm_direct_file_pipeline_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_DIRECT_FILE_PIPELINE");
        }
        assert!(glm_direct_file_pipeline_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_GLM_DIRECT_FILE_PIPELINE", "0");
        }
        assert!(!glm_direct_file_pipeline_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_DIRECT_FILE_PIPELINE");
        }
    }

    #[test]
    fn glm_expert_grouped_defaults_on_for_batch_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_EXPERT_GROUPED");
        }
        assert!(glm_expert_grouped_enabled(61, 488));
        assert!(!glm_expert_grouped_enabled(1, 8));
        assert!(!glm_expert_grouped_enabled(61, 61));

        unsafe {
            std::env::set_var("RNB_CUDA_GLM_EXPERT_GROUPED", "0");
        }
        assert!(!glm_expert_grouped_enabled(61, 488));

        unsafe {
            std::env::remove_var("RNB_CUDA_GLM_EXPERT_GROUPED");
        }
    }

    #[test]
    fn prefill_temp_host_register_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_SLOTS");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_GRANULARITY_KB");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_BYTES");
        }
        assert!(!prefill_temp_host_register_enabled());
        assert_eq!(prefill_temp_host_register_min_slots(), 128);
        assert_eq!(prefill_temp_host_register_granularity_bytes(), 4096);
        assert_eq!(prefill_temp_host_register_min_bytes(), 1024 * 1024);

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER", "1");
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_SLOTS", "64");
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_GRANULARITY_KB", "64");
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_BYTES", "1048576");
        }
        assert!(prefill_temp_host_register_enabled());
        assert_eq!(prefill_temp_host_register_min_slots(), 64);
        assert_eq!(prefill_temp_host_register_granularity_bytes(), 64 * 1024);
        assert_eq!(prefill_temp_host_register_min_bytes(), 1024 * 1024);

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_SLOTS");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_GRANULARITY_KB");
            std::env::remove_var("RNB_CUDA_PREFILL_TEMP_HOST_REGISTER_MIN_BYTES");
        }
    }

    #[test]
    fn mtp_verify_window2_graphs_default_on_and_allow_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_WINDOW2_GRAPHS");
        }
        assert!(mtp_verify_window2_graphs_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_WINDOW2_GRAPHS", "0");
        }
        assert!(!mtp_verify_window2_graphs_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_WINDOW2_GRAPHS");
        }
    }

    #[test]
    fn mtp_verify_gdn_qkv_warp_defaults_on_for_window2_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_GDN_QKV_WARP2");
        }
        assert!(mtp_verify_gdn_qkv_warp_enabled(1));
        assert!(mtp_verify_gdn_qkv_warp_enabled(2));
        assert!(!mtp_verify_gdn_qkv_warp_enabled(3));

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_GDN_QKV_WARP2", "0");
        }
        assert!(mtp_verify_gdn_qkv_warp_enabled(1));
        assert!(!mtp_verify_gdn_qkv_warp_enabled(2));

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_GDN_QKV_WARP2");
        }
    }

    #[test]
    fn mtp_verify_router_stable_key_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_ROUTER_STABLE_KEY");
        }
        assert!(mtp_verify_router_stable_key_enabled());
        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_ROUTER_STABLE_KEY", "0");
        }
        assert!(!mtp_verify_router_stable_key_enabled());
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_ROUTER_STABLE_KEY");
        }
    }

    #[test]
    fn mtp_verify_gdn_stable_keys_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_GDN_STABLE_KEYS");
        }
        assert!(mtp_verify_gdn_stable_keys_enabled());
        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_GDN_STABLE_KEYS", "0");
        }
        assert!(!mtp_verify_gdn_stable_keys_enabled());
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_GDN_STABLE_KEYS");
        }
    }

    #[test]
    fn mtp_verify_selected_gate_pair2_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2");
        }
        assert!(mtp_verify_selected_gate_pair2_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2", "0");
        }
        assert!(!mtp_verify_selected_gate_pair2_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2");
        }
    }

    #[test]
    fn mtp_verify_selected_gate_pair2_silu_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2_SILU");
        }
        assert!(mtp_verify_selected_gate_pair2_silu_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2_SILU", "0");
        }
        assert!(!mtp_verify_selected_gate_pair2_silu_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_PAIR2_SILU");
        }
    }

    #[test]
    fn mtp_verify_selected_down_pair2_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_PAIR2");
        }
        assert!(mtp_verify_selected_down_pair2_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_PAIR2", "0");
        }
        assert!(!mtp_verify_selected_down_pair2_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_DOWN_PAIR2");
        }
    }

    #[test]
    fn mtp_verify_selected_pair_map_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_PAIR_MAP");
        }
        assert!(mtp_verify_selected_pair_map_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_PAIR_MAP", "0");
        }
        assert!(!mtp_verify_selected_pair_map_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_PAIR_MAP");
        }
    }

    #[test]
    fn mtp_verify_selected_gate_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8");
        }
        assert!(mtp_verify_selected_gate_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8", "0");
        }
        assert!(!mtp_verify_selected_gate_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP8");
        }
    }

    #[test]
    fn mtp_verify_selected_gate_warp_reduce_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP_REDUCE");
        }
        assert!(mtp_verify_selected_gate_warp_reduce_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP_REDUCE", "0");
        }
        assert!(!mtp_verify_selected_gate_warp_reduce_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_VERIFY_SELECTED_GATE_WARP_REDUCE");
        }
    }

    #[test]
    fn q6k_argmax_batched_single_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_ARGMAX_BATCHED_SINGLE");
        }
        assert!(q6k_argmax_batched_single_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_ARGMAX_BATCHED_SINGLE", "0");
        }
        assert!(!q6k_argmax_batched_single_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_ARGMAX_BATCHED_SINGLE");
        }
    }

    #[test]
    fn cubin_modules_default_on_and_allow_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_CUBIN_MODULES");
        }
        assert!(cubin_modules_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_CUBIN_MODULES", "0");
        }
        assert!(!cubin_modules_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_CUBIN_MODULES");
        }
    }

    #[test]
    fn qwen35_q4_gate_up_silu_fused_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED");
        }
        assert!(qwen35_q4_gate_up_silu_fused_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED", "1");
        }
        assert!(qwen35_q4_gate_up_silu_fused_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED", "0");
        }
        assert!(!qwen35_q4_gate_up_silu_fused_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_SILU_FUSED");
        }
    }

    #[test]
    fn qwen35_q4_gate_up_q8dot_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_Q4_DOWN");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32");
        }
        assert!(qwen35_q4_gate_up_q8dot_enabled());
        assert!(qwen35_q4_gate_up_q8dot_q4_down_enabled());
        assert!(qwen35_q4_down_q8dot_enabled());
        assert!(qwen35_q4_gate_up_q8dot_mmq_enabled());
        assert!(qwen35_q4_gate_up_q8dot_mmq_group16_enabled());
        assert!(qwen35_q4_gate_up_q8dot_mmq_group32_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_Q4_DOWN", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32", "0");
        }
        assert!(!qwen35_q4_gate_up_q8dot_enabled());
        assert!(!qwen35_q4_gate_up_q8dot_q4_down_enabled());
        assert!(!qwen35_q4_down_q8dot_enabled());
        assert!(!qwen35_q4_gate_up_q8dot_mmq_enabled());
        assert!(!qwen35_q4_gate_up_q8dot_mmq_group16_enabled());
        assert!(!qwen35_q4_gate_up_q8dot_mmq_group32_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_Q4_DOWN");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_DOWN_Q8DOT");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP16");
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8DOT_MMQ_GROUP32");
        }
    }

    #[test]
    fn qwen35_selected_base_stream_defaults_on_for_long_prefill_and_allows_opt_out() {
        let key = "RNB_CUDA_QWEN35_SELECTED_BASE_STREAM";
        unsafe {
            std::env::remove_var(key);
        }
        assert!(!qwen35_selected_base_stream_enabled(31));
        assert!(qwen35_selected_base_stream_enabled(32));
        unsafe {
            std::env::set_var(key, "0");
        }
        assert!(!qwen35_selected_base_stream_enabled(32));
        unsafe {
            std::env::set_var(key, "1");
        }
        assert!(qwen35_selected_base_stream_enabled(31));
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn decode_attention_kv_cache_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_KV_CACHE");
        }
        assert!(decode_attention_kv_cache_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_KV_CACHE", "0");
        }
        assert!(!decode_attention_kv_cache_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_KV_CACHE");
        }
    }

    #[test]
    fn decode_attention_sliding_window_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_SLIDING_WINDOW");
        }
        assert!(!decode_attention_sliding_window_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_SLIDING_WINDOW", "1");
        }
        assert!(decode_attention_sliding_window_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_SLIDING_WINDOW");
        }
    }

    #[test]
    fn decode_attention_hd512_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512");
        }
        assert!(decode_attention_hd512_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD512", "0");
        }
        assert!(!decode_attention_hd512_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512");
        }
    }

    #[test]
    fn decode_attention_hd256_split_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK");
        }
        assert!(decode_attention_hd256_split_enabled());
        assert_eq!(decode_attention_hd256_split_chunk_size(), 256);

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT", "0");
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK", "128");
        }
        assert!(!decode_attention_hd256_split_enabled());
        assert_eq!(decode_attention_hd256_split_chunk_size(), 128);

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK", "7");
        }
        assert_eq!(decode_attention_hd256_split_chunk_size(), 256);

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK");
        }
    }

    #[test]
    fn mtp_verify_attention_hd256_split_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_QUERY_TILE");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_MMA_STREAM_K");
        }
        assert!(mtp_verify_attention_hd256_split_enabled());
        assert_eq!(mtp_verify_attention_hd256_split_chunk_size(), 256);
        assert!(mtp_verify_attention_hd256_query_tile_enabled());
        assert_eq!(
            mtp_verify_attention_hd256_mma_stream_k_enabled(),
            compiled_ampere_mma_supported()
        );
        assert!(!cuda_arch_supports_ampere_mma("sm_75"));
        assert!(cuda_arch_supports_ampere_mma("sm_80"));
        assert!(cuda_arch_supports_ampere_mma("sm_86"));

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT", "0");
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK", "128");
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_QUERY_TILE", "0");
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_MMA_STREAM_K", "0");
        }
        assert!(!mtp_verify_attention_hd256_split_enabled());
        assert_eq!(mtp_verify_attention_hd256_split_chunk_size(), 128);
        assert!(!mtp_verify_attention_hd256_query_tile_enabled());
        assert!(!mtp_verify_attention_hd256_mma_stream_k_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK", "7");
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK", "512");
        }
        assert_eq!(mtp_verify_attention_hd256_split_chunk_size(), 512);

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_SPLIT_CHUNK");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD256_SPLIT_CHUNK");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_QUERY_TILE");
            std::env::remove_var("RNB_CUDA_MTP_ATTN_HD256_MMA_STREAM_K");
        }
    }

    #[test]
    fn decode_attention_hd512_split_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK");
        }
        assert!(decode_attention_hd512_split_enabled());
        assert_eq!(decode_attention_hd512_split_chunk_size(), 512);

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT", "0");
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK", "128");
        }
        assert!(!decode_attention_hd512_split_enabled());
        assert_eq!(decode_attention_hd512_split_chunk_size(), 128);

        unsafe {
            std::env::set_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK", "7");
        }
        assert_eq!(decode_attention_hd512_split_chunk_size(), 512);

        unsafe {
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT");
            std::env::remove_var("RNB_CUDA_DECODE_ATTN_HD512_SPLIT_CHUNK");
        }
    }

    #[test]
    fn prefill_flash_hd512_w256_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256");
        }
        assert!(prefill_flash_attention_hd512_w256_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256", "0");
        }
        assert!(!prefill_flash_attention_hd512_w256_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_FLASH_ATTN_HD512_W256");
        }
    }

    #[test]
    fn prefill_down_copy_overlap_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP");
        }
        assert!(!prefill_down_copy_overlap_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP", "1");
        }
        assert!(prefill_down_copy_overlap_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_DOWN_COPY_OVERLAP");
        }
    }

    #[test]
    fn prefill_moe_full_layer_stays_quarantined_after_xid79() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER");
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY");
            std::env::remove_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS");
        }
        assert!(!prefill_moe_full_layer_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER", "1");
        }
        assert!(!prefill_moe_full_layer_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY", "1");
        }
        assert!(!prefill_moe_full_layer_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS", "1");
        }
        assert!(prefill_moe_full_layer_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER");
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_FULL_LAYER_UNSAFE_RETRY");
            std::env::remove_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS");
        }
    }

    #[test]
    fn qwen35_full_layer_device_slot_ptrs_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS");
        }
        assert!(!qwen35_full_layer_device_slot_ptrs_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS", "1");
        }
        assert!(qwen35_full_layer_device_slot_ptrs_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_FULL_LAYER_DEVICE_SLOT_PTRS");
        }
    }

    #[test]
    fn prefill_delta_k128_warp4_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_DELTA_K128_WARP4");
        }
        assert!(prefill_delta_k128_warp4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_DELTA_K128_WARP4", "0");
        }
        assert!(!prefill_delta_k128_warp4_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_DELTA_K128_WARP4");
        }
    }

    #[test]
    fn group4_down_row8_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_GROUP4_DOWN_ROW8");
        }
        assert!(!group4_down_row8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_GROUP4_DOWN_ROW8", "1");
        }
        assert!(group4_down_row8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_GROUP4_DOWN_ROW8");
        }
    }

    #[test]
    fn qwen35_q4_down_group4_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4");
        }
        assert!(qwen35_q4_down_group4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", "1");
        }
        assert!(qwen35_q4_down_group4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4", "0");
        }
        assert!(!qwen35_q4_down_group4_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_DOWN_GROUP4");
        }
    }

    #[test]
    fn q6k_output_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_OUTPUT_WARP8");
        }
        assert!(q6k_output_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_OUTPUT_WARP8", "0");
        }
        assert!(!q6k_output_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_OUTPUT_WARP8");
        }
    }

    #[test]
    fn q6k_argmax_gpu_reduce_defaults_on_for_large_outputs() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE");
        }
        assert!(!q6k_fused_argmax_gpu_reduce_enabled(4096));
        assert!(q6k_fused_argmax_gpu_reduce_enabled(8192));
        assert!(q6k_fused_argmax_gpu_reduce_enabled(248_320));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE", "0");
        }
        assert!(!q6k_fused_argmax_gpu_reduce_enabled(248_320));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE", "1");
        }
        assert!(q6k_fused_argmax_gpu_reduce_enabled(4096));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_FUSED_ARGMAX_GPU_REDUCE");
        }
    }

    #[test]
    fn packed_long_gemv_warp4_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_PACKED_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_Q6K_PACKED_GEMV_WARP4");
        }
        assert!(!q4k_packed_gemv_warp4_enabled());
        assert!(!q6k_packed_gemv_warp4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PACKED_GEMV_WARP4", "1");
            std::env::set_var("RNB_CUDA_Q6K_PACKED_GEMV_WARP4", "1");
        }
        assert!(q4k_packed_gemv_warp4_enabled());
        assert!(q6k_packed_gemv_warp4_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_PACKED_GEMV_WARP4");
            std::env::remove_var("RNB_CUDA_Q6K_PACKED_GEMV_WARP4");
        }
    }

    #[test]
    fn q4k_batch_raw_seq4_defaults_on_for_long_prefill_and_allows_overrides() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4");
        }
        assert!(q4k_batch_raw_seq4_enabled(1115, 2560, 10));
        assert!(q4k_batch_raw_seq4_enabled(1115, 256, 26));
        assert!(!q4k_batch_raw_seq4_enabled(63, 256, 26));
        assert!(!q4k_batch_raw_seq4_enabled(4, 2560, 10));
        assert!(!q4k_batch_raw_seq4_enabled(1115, 32, 10));

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", "0");
        }
        assert!(!q4k_batch_raw_seq4_enabled(1115, 2560, 10));

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4", "1");
        }
        assert!(q4k_batch_raw_seq4_enabled(4, 2560, 10));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_BATCH_RAW_SEQ4");
        }
    }

    #[test]
    fn q4k_mmq_tile32_defaults_on_for_long_prefill_and_allows_opt_out() {
        // 공용 잠금 — 이 env 들은 런타임 Q4 계약 테스트와 공유한다.
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_MMQ_TILE32");
        }
        assert!(q4k_mmq_tile32_enabled(1115, 2560, 10));
        // cu219: 기본 min_seq 는 8 (partial tile 마스킹으로 안전, 27B 15/100
        // −14.5% 근거). env 로 경계를 되돌릴 수 있다.
        assert!(q4k_mmq_tile32_enabled(8, 2560, 10));
        assert!(!q4k_mmq_tile32_enabled(7, 2560, 10));
        assert!(!q4k_mmq_tile32_enabled(1115, 512, 10));
        assert!(!q4k_mmq_tile32_enabled(1115, 2560, 3));
        unsafe {
            std::env::set_var("RNB_CUDA_MMQ_TILE32_MIN_SEQ", "32");
        }
        assert!(!q4k_mmq_tile32_enabled(31, 2560, 10));
        assert!(q4k_mmq_tile32_enabled(32, 2560, 10));
        unsafe {
            std::env::remove_var("RNB_CUDA_MMQ_TILE32_MIN_SEQ");
        }

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_MMQ_TILE32", "0");
        }
        assert!(!q4k_mmq_tile32_enabled(1115, 2560, 10));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_MMQ_TILE32");
        }
    }

    #[test]
    fn mmq_tile_seq64_defaults_on_at_64_and_allows_opt_out() {
        // 공용 잠금 — 이 env(RNB_CUDA_MMQ_TILE_SEQ64)는 계약 테스트도 쓴다.
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_MMQ_TILE_SEQ64");
        }
        assert!(mmq_tile_seq64_enabled(64));
        assert!(mmq_tile_seq64_enabled(1139));
        assert!(!mmq_tile_seq64_enabled(63));

        unsafe {
            std::env::set_var("RNB_CUDA_MMQ_TILE_SEQ64", "0");
        }
        assert!(!mmq_tile_seq64_enabled(1139));

        unsafe {
            std::env::remove_var("RNB_CUDA_MMQ_TILE_SEQ64");
        }
    }

    #[test]
    fn q6k_mmq_tile32_defaults_on_for_long_prefill_and_allows_opt_out() {
        // 공용 잠금 — 이 env 들은 런타임 Q6 계약 테스트와 공유한다.
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_MMQ_TILE32");
        }
        assert!(q6k_mmq_tile32_enabled(1115, 8192, 8));
        assert!(q6k_mmq_tile32_enabled(8, 8192, 8));
        assert!(!q6k_mmq_tile32_enabled(7, 8192, 8));
        assert!(!q6k_mmq_tile32_enabled(1115, 512, 8));
        assert!(!q6k_mmq_tile32_enabled(1115, 8192, 3));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_MMQ_TILE32", "0");
        }
        assert!(!q6k_mmq_tile32_enabled(1115, 8192, 8));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_MMQ_TILE32");
        }
    }

    #[test]
    fn q4k_mmq_tile64_requires_seq64_and_row_tile_and_allows_opt_out() {
        // 공용 잠금 — 이 env(RNB_CUDA_Q4K_MMQ_TILE64)는 계약 테스트도 쓴다.
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_MMQ_TILE_SEQ64");
            std::env::remove_var("RNB_CUDA_Q4K_MMQ_TILE64");
        }
        assert!(q4k_mmq_tile64_enabled(64, 64));
        assert!(q4k_mmq_tile64_enabled(1139, 12288));
        assert!(!q4k_mmq_tile64_enabled(63, 12288));
        assert!(!q4k_mmq_tile64_enabled(1139, 63));

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_MMQ_TILE64", "0");
        }
        assert!(!q4k_mmq_tile64_enabled(1139, 12288));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_MMQ_TILE64");
        }
    }

    #[test]
    fn q5k_q6k_mmq_tile64_share_seq64_requirement_and_own_opt_out() {
        // 공용 잠금 — 이 env 들은 계약 테스트도 쓴다.
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_MMQ_TILE_SEQ64");
            std::env::remove_var("RNB_CUDA_Q5K_MMQ_TILE64");
            std::env::remove_var("RNB_CUDA_Q6K_MMQ_TILE64");
        }
        assert!(q5k_mmq_tile64_enabled(64, 64));
        assert!(q5k_mmq_tile64_enabled(1139, 12288));
        assert!(q6k_mmq_tile64_enabled(64, 64));
        assert!(q6k_mmq_tile64_enabled(1139, 12288));
        assert!(!q5k_mmq_tile64_enabled(63, 12288));
        assert!(!q6k_mmq_tile64_enabled(1139, 63));

        unsafe {
            std::env::set_var("RNB_CUDA_Q5K_MMQ_TILE64", "0");
        }
        assert!(!q5k_mmq_tile64_enabled(1139, 12288));
        assert!(q6k_mmq_tile64_enabled(1139, 12288));
        unsafe {
            std::env::remove_var("RNB_CUDA_Q5K_MMQ_TILE64");
        }

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_MMQ_TILE64", "0");
        }
        assert!(!q6k_mmq_tile64_enabled(1139, 12288));
        assert!(q5k_mmq_tile64_enabled(1139, 12288));
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_MMQ_TILE64");
        }
    }

    #[test]
    fn q8_0_mmq_tile32_defaults_on_for_long_prefill_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_MMQ_TILE32");
        }
        assert!(q8_0_mmq_tile32_enabled(1139, 2048, 64));
        assert!(!q8_0_mmq_tile32_enabled(31, 2048, 64));
        assert!(!q8_0_mmq_tile32_enabled(1139, 127, 64));
        assert!(!q8_0_mmq_tile32_enabled(1139, 2048, 3));

        unsafe {
            std::env::set_var("RNB_CUDA_Q8_0_MMQ_TILE32", "0");
        }
        assert!(!q8_0_mmq_tile32_enabled(1139, 2048, 64));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_MMQ_TILE32");
        }
    }

    #[test]
    fn qwen35_q5_down_q8dot_mmq_is_prefill_only_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ");
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP32");
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP64");
        }
        assert!(qwen35_q5_down_q8dot_mmq_enabled(1139));
        assert!(qwen35_q5_down_q8dot_mmq_enabled(32));
        assert!(!qwen35_q5_down_q8dot_mmq_enabled(31));
        assert!(!qwen35_q5_down_q8dot_mmq_enabled(2));
        assert!(qwen35_q5_down_q8dot_mmq_group32_enabled());
        assert!(qwen35_q5_down_q8dot_mmq_group64_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP32", "0");
            std::env::set_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP64", "0");
        }
        assert!(!qwen35_q5_down_q8dot_mmq_enabled(1139));
        assert!(!qwen35_q5_down_q8dot_mmq_group32_enabled());
        assert!(!qwen35_q5_down_q8dot_mmq_group64_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ");
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP32");
            std::env::remove_var("RNB_CUDA_QWEN35_Q5_DOWN_Q8DOT_MMQ_GROUP64");
        }
    }

    #[test]
    fn qwen35_q4_gate_up_q8_handoff_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF");
        }
        assert!(qwen35_q4_gate_up_q8_handoff_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF", "0");
        }
        assert!(!qwen35_q4_gate_up_q8_handoff_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_Q4_GATE_UP_Q8_HANDOFF");
        }
    }

    #[test]
    fn q6k_packed_batch_warp4_defaults_on_for_mid_blocks_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
        }
        assert!(q6k_packed_batch_warp4_enabled(14));
        assert!(q6k_packed_batch_warp4_enabled(8));
        assert!(!q6k_packed_batch_warp4_enabled(7));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6_PACKED_BATCH_WARP4", "0");
        }
        assert!(!q6k_packed_batch_warp4_enabled(14));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_WARP4");
        }
    }

    #[test]
    fn q6k_packed_batch_seq8_stays_opt_in_after_abab_regression() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8");
        }
        assert!(!q6k_packed_batch_seq8_enabled(8, 8));
        assert!(!q6k_packed_batch_seq8_enabled(7, 8));
        assert!(!q6k_packed_batch_seq8_enabled(8, 7));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8", "1");
        }
        assert!(q6k_packed_batch_seq8_enabled(1115, 14));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ8");
        }
    }

    #[test]
    fn q4k_prefill_f16_split_flags_follow_global_and_allow_overrides() {
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ");
        }
        assert!(!prefill_q4k_f16_gemm_enabled());
        assert!(!prefill_q4k_f16_qkv_gemm_enabled());
        assert!(!prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM", "1");
        }
        assert!(!prefill_q4k_f16_gemm_enabled());
        assert!(!prefill_q4k_f16_qkv_gemm_enabled());
        assert!(!prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
        }
        assert!(prefill_q4k_f16_gemm_enabled());
        assert!(prefill_q4k_f16_qkv_gemm_enabled());
        assert!(prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM", "0");
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "1");
        }
        assert!(prefill_q4k_f16_gemm_enabled());
        assert!(!prefill_q4k_f16_qkv_gemm_enabled());
        assert!(prefill_q4k_f16_o_proj_enabled());
        assert!(!prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ", "force");
        }
        assert!(prefill_q4k_f16_o_proj_enabled());
        assert!(prefill_q4k_f16_o_proj_force_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_QKV_GEMM");
            std::env::remove_var("RNB_CUDA_Q4K_PREFILL_F16_O_PROJ");
        }
    }

    #[test]
    fn q8_output_q8dot_argmax_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX");
        }
        assert!(!q8_0_output_q8dot_argmax_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX", "1");
        }
        assert!(q8_0_output_q8dot_argmax_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX");
        }
    }

    #[test]
    fn dense_expert_graph_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_DENSE_EXPERT_GRAPH");
        }
        assert!(!dense_expert_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_DENSE_EXPERT_GRAPH", "1");
        }
        assert!(dense_expert_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_DENSE_EXPERT_GRAPH");
        }
    }

    #[test]
    fn cu69_dense_chain_graph_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CU69_DENSE_CHAIN_GRAPH");
        }
        assert!(!cu69_dense_chain_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CU69_DENSE_CHAIN_GRAPH", "1");
        }
        assert!(cu69_dense_chain_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CU69_DENSE_CHAIN_GRAPH");
        }
    }

    #[test]
    fn cu69_dense_chain_graph_trace_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CU69_DENSE_CHAIN_GRAPH_TRACE");
        }
        assert!(!cu69_dense_chain_graph_trace_enabled());

        unsafe {
            std::env::set_var("RNB_CU69_DENSE_CHAIN_GRAPH_TRACE", "1");
        }
        assert!(cu69_dense_chain_graph_trace_enabled());

        unsafe {
            std::env::remove_var("RNB_CU69_DENSE_CHAIN_GRAPH_TRACE");
        }
    }

    #[test]
    fn cu71_layer_segment_graph_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CU71_LAYER_SEGMENT_GRAPH");
        }
        assert!(!cu71_layer_segment_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CU71_LAYER_SEGMENT_GRAPH", "1");
        }
        assert!(cu71_layer_segment_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CU71_LAYER_SEGMENT_GRAPH");
        }
    }

    #[test]
    fn cu71_layer_segment_graph_trace_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CU71_LAYER_SEGMENT_GRAPH_TRACE");
        }
        assert!(!cu71_layer_segment_graph_trace_enabled());

        unsafe {
            std::env::set_var("RNB_CU71_LAYER_SEGMENT_GRAPH_TRACE", "1");
        }
        assert!(cu71_layer_segment_graph_trace_enabled());

        unsafe {
            std::env::remove_var("RNB_CU71_LAYER_SEGMENT_GRAPH_TRACE");
        }
    }

    #[test]
    fn qwen35_decode_moe_graph_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MOE_GRAPH");
        }
        assert!(qwen35_decode_moe_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MOE_GRAPH", "0");
        }
        assert!(!qwen35_decode_moe_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MOE_GRAPH");
        }
    }

    #[test]
    fn qwen35_selected_sparse_compound_graph_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH");
        }
        assert!(qwen35_selected_sparse_compound_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH", "0");
        }
        assert!(!qwen35_selected_sparse_compound_graph_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH", "1");
        }
        assert!(qwen35_selected_sparse_compound_graph_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH");
        }
    }

    #[test]
    fn qwen35_selected_sparse_compound_graph_zero_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO");
        }
        assert!(!qwen35_selected_sparse_compound_graph_zero_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO", "0");
        }
        assert!(!qwen35_selected_sparse_compound_graph_zero_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO", "1");
        }
        assert!(qwen35_selected_sparse_compound_graph_zero_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_GRAPH_ZERO");
        }
    }

    #[test]
    fn q8_0_gemv_warp4_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP4");
        }
        assert!(q8_0_gemv_warp4_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q8_0_GEMV_WARP4", "0");
        }
        assert!(!q8_0_gemv_warp4_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP4");
        }
    }

    #[test]
    fn q8_0_gemv_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP8");
        }
        assert!(q8_0_gemv_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q8_0_GEMV_WARP8", "0");
        }
        assert!(!q8_0_gemv_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q8_0_GEMV_WARP8");
        }
    }

    #[test]
    fn q4k_gemv_batch_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_GEMV_BATCH_WARP8");
        }
        assert!(q4k_gemv_batch_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q4K_GEMV_BATCH_WARP8", "0");
        }
        assert!(!q4k_gemv_batch_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q4K_GEMV_BATCH_WARP8");
        }
    }

    #[test]
    fn q6k_gemv_batch_warp8_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_GEMV_BATCH_WARP8");
        }
        assert!(q6k_gemv_batch_warp8_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_GEMV_BATCH_WARP8", "0");
        }
        assert!(!q6k_gemv_batch_warp8_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_GEMV_BATCH_WARP8");
        }
    }

    #[test]
    fn q6k_gemv_batch_seq4_defaults_on_for_long_narrow_prefill_and_allows_overrides() {
        let _guard = crate::runtime::cuda_test_env_lock();
        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_GEMV_BATCH_SEQ4_WARP8");
        }
        assert!(q6k_gemv_batch_seq4_warp8_enabled(1144, 256, 26));
        assert!(!q6k_gemv_batch_seq4_warp8_enabled(63, 256, 26));
        assert!(!q6k_gemv_batch_seq4_warp8_enabled(1144, 32, 26));
        assert!(!q6k_gemv_batch_seq4_warp8_enabled(1144, 256, 3));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_GEMV_BATCH_SEQ4_WARP8", "0");
        }
        assert!(!q6k_gemv_batch_seq4_warp8_enabled(1144, 256, 26));

        unsafe {
            std::env::set_var("RNB_CUDA_Q6K_GEMV_BATCH_SEQ4_WARP8", "1");
        }
        assert!(q6k_gemv_batch_seq4_warp8_enabled(4, 32, 1));

        unsafe {
            std::env::remove_var("RNB_CUDA_Q6K_GEMV_BATCH_SEQ4_WARP8");
        }
    }

    #[test]
    fn resident_q4k_touch_hits_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS");
        }
        assert!(!resident_q4k_touch_hits_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS", "1");
        }
        assert!(resident_q4k_touch_hits_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_TOUCH_HITS");
        }
    }

    #[test]
    fn resident_q4k_arena_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_ARENA");
        }
        assert!(!resident_q4k_arena_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_ARENA", "1");
        }
        assert!(resident_q4k_arena_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_ARENA");
        }
    }

    #[test]
    fn resident_q4k_batch_pinned_staging_scales_with_batch_size() {
        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES");
        }
        assert!(!resident_q4k_batch_pinned_staging_enabled(
            2 * 1024 * 1024,
            1
        ));
        assert!(!resident_q4k_batch_pinned_staging_enabled(1024 * 1024, 2));
        assert!(resident_q4k_batch_pinned_staging_enabled(
            2 * 1024 * 1024,
            2
        ));

        unsafe {
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED", "0");
        }
        assert!(!resident_q4k_batch_pinned_staging_enabled(
            16 * 1024 * 1024,
            8
        ));

        unsafe {
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED", "1");
        }
        assert!(resident_q4k_batch_pinned_staging_enabled(1, 1));

        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED");
            std::env::set_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES", "4096");
        }
        assert!(resident_q4k_batch_pinned_staging_enabled(4096, 2));

        unsafe {
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED");
            std::env::remove_var("RNB_CUDA_RESIDENT_Q4K_BATCH_PINNED_MIN_BYTES");
        }
    }

    #[test]
    fn qwen35_decode_q4k_arena_defaults_on_and_allows_opt_out() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA");
        }
        assert!(qwen35_decode_q4k_arena_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA", "0");
        }
        assert!(!qwen35_decode_q4k_arena_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_Q4K_ARENA");
        }
    }

    #[test]
    fn qwen35_decode_resident_batch_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_RESIDENT_BATCH");
        }
        assert!(!qwen35_decode_resident_batch_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_DECODE_RESIDENT_BATCH", "1");
        }
        assert!(qwen35_decode_resident_batch_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_DECODE_RESIDENT_BATCH");
        }
    }

    #[test]
    fn qwen35_prefill_hot_resident_defaults_off_and_caps_auto_budget() {
        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT");
            std::env::remove_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MB");
        }
        assert!(!qwen35_prefill_hot_resident_enabled());
        assert_eq!(
            qwen35_prefill_hot_resident_budget_bytes(8 * 1024 * 1024 * 1024),
            16 * 1024 * 1024
        );

        unsafe {
            std::env::set_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT", "1");
            std::env::set_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MB", "6");
        }
        assert!(qwen35_prefill_hot_resident_enabled());
        assert_eq!(
            qwen35_prefill_hot_resident_budget_bytes(8 * 1024 * 1024 * 1024),
            6 * 1024 * 1024
        );

        unsafe {
            std::env::remove_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT");
            std::env::remove_var("RNB_CUDA_QWEN35_PREFILL_HOT_RESIDENT_MB");
        }
    }

    #[test]
    fn prefill_moe_sync_before_sparse_stays_on_after_xid79() {
        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_SYNC_BEFORE_SPARSE");
        }
        assert!(prefill_moe_sync_before_sparse_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_PREFILL_MOE_SYNC_BEFORE_SPARSE", "0");
        }
        assert!(prefill_moe_sync_before_sparse_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_PREFILL_MOE_SYNC_BEFORE_SPARSE");
        }
    }

    #[test]
    fn mtp_expert_trace_defaults_off_and_allows_opt_in() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_TRACE");
        }
        assert!(!mtp_expert_trace_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_TRACE", "1");
        }
        assert!(mtp_expert_trace_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_TRACE");
        }
    }

    #[test]
    fn mtp_expert_hot_resident_follows_device_verify_and_allows_override() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT");
            std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        }
        assert!(!mtp_expert_hot_resident_enabled());

        unsafe {
            std::env::set_var("RNB_MTP_DEVICE_VERIFY", "1");
        }
        assert!(mtp_expert_hot_resident_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT", "0");
        }
        assert!(!mtp_expert_hot_resident_enabled());

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT", "1");
            std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        }
        assert!(mtp_expert_hot_resident_enabled());

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_HOT_RESIDENT");
            std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        }
    }

    #[test]
    fn mtp_expert_extra_resident_budget_scales_with_cache_limit() {
        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB");
        }
        assert_eq!(
            mtp_expert_extra_resident_budget_bytes(256 * 1024 * 1024),
            1024 * 1024
        );
        assert_eq!(
            mtp_expert_extra_resident_budget_bytes_for_layer(1024 * 1024 * 1024, 0),
            4 * 1024 * 1024
        );
        assert_eq!(
            mtp_expert_extra_resident_budget_bytes_for_layer(1024 * 1024 * 1024, 8),
            8 * 1024 * 1024
        );

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB", "32");
        }
        assert_eq!(
            mtp_expert_extra_resident_budget_bytes(256 * 1024 * 1024),
            32 * 1024 * 1024
        );

        unsafe {
            std::env::set_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT", "0");
        }
        assert_eq!(mtp_expert_extra_resident_budget_bytes(256 * 1024 * 1024), 0);

        unsafe {
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT");
            std::env::remove_var("RNB_CUDA_MTP_EXPERT_EXTRA_RESIDENT_MB");
        }
    }
}
