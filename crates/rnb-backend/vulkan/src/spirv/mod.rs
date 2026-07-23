mod argmax_pairs_stage1;
pub(crate) mod builder;
mod embed_lookup;
mod gdn_delta_sequence;
mod logit_argmax_q4k;
mod logit_argmax_q5k;
mod native_quant;
mod q4k_gate_up;
mod q5k_q8k;
mod q8_q8k;
pub use argmax_pairs_stage1::emit_argmax_pairs_f32_stage1;
pub use builder::{
    emit_argmax_pairs_f32, emit_attention_decode, emit_depthwise_conv1d_silu, emit_elem_add,
    emit_elem_add_broadcast, emit_elem_add_out, emit_embed_lookup_q6k, emit_f32_gemv,
    emit_gdn_delta_precompute, emit_gdn_delta_sequence, emit_gdn_delta_step,
    emit_gdn_gated_norm_silu, emit_kv_append, emit_logit_argmax_q6k, emit_logit_argmax_q8_0,
    emit_logit_argmax_q8_0_chunked, emit_q4k_block_reduce, emit_q4k_gemv, emit_q4k_gemv_batch4,
    emit_q4k_gemv_block_partial, emit_q4k_gemv_rowmajor, emit_q4k_gemv_rowmajor_batched,
    emit_q4k_gemv_wg_reduce, emit_q4k_q8k_gemv, emit_q5k_gemv, emit_q5k_gemv_batch2,
    emit_q5k_gemv_batch4, emit_q6k_gemv, emit_q6k_gemv_batch4, emit_q6k_gemv_batch4_f16,
    emit_q6k_q8k_gemv, emit_q8_0_gemv, emit_quantize_to_q8k, emit_rms_norm, emit_rope_apply,
    emit_sigmoid_mul, emit_silu_mul, emit_split_gated_q, Id, SpirvModule,
};
pub use embed_lookup::{emit_embed_lookup_q4k, emit_embed_lookup_q5k, emit_embed_lookup_q8_0};
pub use gdn_delta_sequence::emit_gdn_delta_sequence_d128;
pub use logit_argmax_q4k::emit_logit_argmax_q4k;
pub use logit_argmax_q5k::emit_logit_argmax_q5k;
pub use native_quant::{
    emit_native_quant_embed_lookup, emit_native_quant_gemv, emit_native_quant_logit_argmax,
};
pub use q4k_gate_up::emit_q4k_gate_up_batch4;
pub use q5k_q8k::emit_q5k_q8k_gemv;
pub use q8_q8k::emit_q8_q8k_gemv;
