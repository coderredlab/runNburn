pub(crate) mod builder;
pub use builder::{
    emit_argmax_pairs_f32, emit_attention_decode, emit_depthwise_conv1d_silu, emit_elem_add,
    emit_elem_add_out, emit_embed_lookup_q6k, emit_f32_gemv, emit_gdn_delta_step,
    emit_gdn_gated_norm_silu, emit_kv_append, emit_logit_argmax_q6k, emit_logit_argmax_q8_0,
    emit_logit_argmax_q8_0_chunked, emit_q4k_block_reduce, emit_q4k_gemv,
    emit_q4k_gemv_block_partial, emit_q4k_gemv_rowmajor, emit_q4k_gemv_wg_reduce,
    emit_q4k_q8k_gemv, emit_q5k_gemv, emit_q6k_gemv, emit_q6k_q8k_gemv, emit_q8_0_gemv,
    emit_quantize_to_q8k, emit_rms_norm, emit_rope_apply, emit_sigmoid_mul, emit_silu_mul,
    emit_split_gated_q, Id, SpirvModule,
};
