fn main() {
    let model_path =
        std::env::var("RNB_MODEL").unwrap_or_else(|_| "models/gemma-4-E2B.Q4_K_M.gguf".to_string());
    let model = rnb_loader::load_model(&std::path::PathBuf::from(&model_path)).unwrap();

    let num_layers = model.metadata.num_layers;
    let num_heads = model.metadata.num_heads;
    let num_kv_heads = model.metadata.num_kv_heads;
    let meta_head_dim = model.metadata.head_dim;

    println!(
        "meta: layers={} heads={} kv_heads={} head_dim={}",
        num_layers, num_heads, num_kv_heads, meta_head_dim
    );

    for i in 0..num_layers {
        let q = model.float_shapes.get(&format!("blk.{i}.attn_q.weight"));
        let k = model.float_shapes.get(&format!("blk.{i}.attn_k.weight"));
        let v = model.float_shapes.get(&format!("blk.{i}.attn_v.weight"));
        let o = model
            .float_shapes
            .get(&format!("blk.{i}.attn_output.weight"));

        if let (Some(q), Some(k), Some(v), Some(o)) = (q, k, v, o) {
            let q_rows = q[0];
            let k_rows = k[0];
            let v_rows = v[0];
            let o_rows = o[0];
            let q_head_dim = q_rows / num_heads;
            let kv_head_dim = k_rows / num_kv_heads;
            println!(
                "layer={i:02} q_rows={q_rows} k_rows={k_rows} v_rows={v_rows} o_rows={o_rows} q_head_dim={q_head_dim} kv_head_dim={kv_head_dim}"
            );
        }
    }
}
