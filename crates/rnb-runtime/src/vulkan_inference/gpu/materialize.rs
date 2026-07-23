use super::*;

pub fn write_gdn_conv_state_f32(
    runtime: &mut Runtime,
    layer_idx: usize,
    conv_state: &[f32],
) -> Result<(), String> {
    runtime.write_gdn_conv_state_f32_for_layer(layer_idx, conv_state)
}

pub fn materialize_gdn_conv_state_f32(
    runtime: &mut Runtime,
    layer_idx: usize,
    conv_state: &mut [f32],
) -> Result<(), String> {
    runtime.materialize_gdn_conv_state_f32_for_layer(layer_idx, conv_state)
}

pub fn materialize_attention_kv(
    runtime: &mut Runtime,
    request: AttentionKvMaterializeRequest,
) -> Result<(Vec<u16>, Vec<u16>), String> {
    if request.num_kv_heads() == 1 {
        runtime.materialize_attention_kv_f16_for_layer(
            request.layer_idx(),
            request.total_tokens() * request.kv_dim(),
        )
    } else {
        runtime.materialize_attention_kv_f16_grouped_for_layer(
            request.layer_idx(),
            request.num_kv_heads(),
            request.total_tokens() * request.head_dim(),
            request.head_dim(),
        )
    }
}

pub fn materialize_attention_kv_range_untracked(
    runtime: &mut Runtime,
    request: AttentionKvMaterializeRangeRequest,
) -> Result<((Vec<u16>, Vec<u16>), usize), String> {
    if request.num_kv_heads() == 1 {
        runtime.materialize_attention_kv_f16_range_for_layer_untracked(
            request.layer_idx(),
            request.pos_start() * request.head_dim(),
            request.kv_len() * request.head_dim(),
        )
    } else {
        runtime.materialize_attention_kv_f16_grouped_range_for_layer_untracked(
            request.layer_idx(),
            request.num_kv_heads(),
            request.pos_start(),
            request.kv_len(),
            request.head_dim(),
        )
    }
}
