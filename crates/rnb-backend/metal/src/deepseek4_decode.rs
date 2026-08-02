use super::*;

pub(crate) struct QFrontCarrier {
    hidden_dim: usize,
    q_rank: usize,
    output_layout: Vec<(usize, usize)>,
    input: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    outputs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    output_rows: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    hidden: Retained<ProtocolObject<dyn MTLBuffer>>,
    rank: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_a_weight_offset: Retained<ProtocolObject<dyn MTLBuffer>>,
    zero_offset: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl QFrontCarrier {
    pub(crate) fn new(
        ctx: &compute::MetalContext,
        hidden_dim: usize,
        q_rank: usize,
        output_layout: &[(usize, usize)],
    ) -> Self {
        debug_assert!(output_layout.iter().all(|&(_, cols)| cols == q_rank));
        Self {
            hidden_dim,
            q_rank,
            output_layout: output_layout.to_vec(),
            input: ffn_chain::empty_f32_buf(ctx, hidden_dim),
            q_a: ffn_chain::empty_f32_buf(ctx, q_rank),
            q_norm: ffn_chain::empty_f32_buf(ctx, q_rank),
            outputs: output_layout
                .iter()
                .map(|&(rows, _)| ffn_chain::empty_f32_buf(ctx, rows))
                .collect(),
            output_rows: output_layout
                .iter()
                .map(|&(rows, _)| ffn_chain::u32_buf(ctx, rows as u32))
                .collect(),
            hidden: ffn_chain::u32_buf(ctx, hidden_dim as u32),
            rank: ffn_chain::u32_buf(ctx, q_rank as u32),
            q_a_weight_offset: ffn_chain::u32_buf(ctx, 0),
            zero_offset: ffn_chain::u32_buf(ctx, 0),
            eps: ffn_chain::f32_buf(ctx, 0.0),
        }
    }

    fn store_scalar<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, value: T) {
        unsafe {
            std::ptr::write(buffer.contents().as_ptr() as *mut T, value);
        }
    }

    pub(crate) fn dispatch(
        &self,
        ctx: &compute::MetalContext,
        q_a_weight: &(Retained<ProtocolObject<dyn MTLBuffer>>, u32),
        q_norm_weight: &(Retained<ProtocolObject<dyn MTLBuffer>>, u32),
        output_weights: &[(Retained<ProtocolObject<dyn MTLBuffer>>, u32)],
        input: &[f32],
        eps: f32,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        assert_eq!(input.len(), self.hidden_dim);
        assert_eq!(output_weights.len(), self.output_layout.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input.contents().as_ptr() as *mut f32,
                input.len(),
            );
        }
        Self::store_scalar(&self.q_a_weight_offset, q_a_weight.1);
        Self::store_scalar(&self.eps, eps);

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("compute encoder");
        compute::encode_gemv_q5k_auto(
            ctx,
            &encoder,
            &q_a_weight.0,
            &self.input,
            &self.q_a,
            &self.rank,
            &self.hidden,
            &self.q_a_weight_offset,
            self.q_rank,
        );
        ffn_chain::encode_rms_norm_at(
            ctx,
            &encoder,
            &self.q_a,
            &q_norm_weight.0,
            q_norm_weight.1 as usize,
            &self.q_norm,
            &self.rank,
            &self.eps,
        );
        for (index, ((weight, output), &(rows, _))) in output_weights
            .iter()
            .zip(&self.outputs)
            .zip(&self.output_layout)
            .enumerate()
        {
            compute::encode_gemv_q8_0_at(
                ctx,
                &encoder,
                &weight.0,
                weight.1 as usize,
                &self.q_norm,
                0,
                output,
                0,
                &self.output_rows[index],
                &self.rank,
                &self.zero_offset,
                rows,
            );
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        let q_norm = unsafe {
            std::slice::from_raw_parts(self.q_norm.contents().as_ptr() as *const f32, self.q_rank)
                .to_vec()
        };
        let outputs = self
            .output_layout
            .iter()
            .zip(&self.outputs)
            .map(|(&(rows, _), output)| unsafe {
                std::slice::from_raw_parts(output.contents().as_ptr() as *const f32, rows).to_vec()
            })
            .collect();
        (q_norm, outputs)
    }
}

pub(crate) struct Q8MultiGemvCarrier {
    layout: Vec<(usize, usize)>,
    input_offsets: Vec<usize>,
    output_offsets: Vec<usize>,
    input: Retained<ProtocolObject<dyn MTLBuffer>>,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
    n: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    k: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    zero_offset: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl Q8MultiGemvCarrier {
    pub(crate) fn new(ctx: &compute::MetalContext, layout: &[(usize, usize)]) -> Self {
        let mut input_elements = 0usize;
        let mut output_elements = 0usize;
        let mut input_offsets = Vec::with_capacity(layout.len());
        let mut output_offsets = Vec::with_capacity(layout.len());
        let mut n = Vec::with_capacity(layout.len());
        let mut k = Vec::with_capacity(layout.len());
        for &(rows, cols) in layout {
            input_offsets.push(input_elements * std::mem::size_of::<f32>());
            output_offsets.push(output_elements * std::mem::size_of::<f32>());
            input_elements += cols;
            output_elements += rows;
            n.push(ffn_chain::u32_buf(ctx, rows as u32));
            k.push(ffn_chain::u32_buf(ctx, cols as u32));
        }
        let shared = MTLResourceOptions::StorageModeShared;
        let input = ctx
            .device
            .newBufferWithLength_options(input_elements * std::mem::size_of::<f32>(), shared)
            .expect("Metal: DeepSeek4 multi-GEMV input buffer");
        let output = ctx
            .device
            .newBufferWithLength_options(output_elements * std::mem::size_of::<f32>(), shared)
            .expect("Metal: DeepSeek4 multi-GEMV output buffer");
        Self {
            layout: layout.to_vec(),
            input_offsets,
            output_offsets,
            input,
            output,
            n,
            k,
            zero_offset: ffn_chain::u32_buf(ctx, 0),
        }
    }

    fn upload_inputs(&self, inputs: &[&[f32]]) {
        for ((&(_, cols), &offset), input) in
            self.layout.iter().zip(&self.input_offsets).zip(inputs)
        {
            assert_eq!(input.len(), cols);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    self.input.contents().as_ptr().add(offset) as *mut f32,
                    cols,
                );
            }
        }
    }

    pub(crate) fn dispatch(
        &self,
        ctx: &compute::MetalContext,
        weights: &[(Retained<ProtocolObject<dyn MTLBuffer>>, usize)],
        inputs: &[&[f32]],
    ) -> Vec<Vec<f32>> {
        assert_eq!(weights.len(), self.layout.len());
        assert_eq!(inputs.len(), self.layout.len());
        self.upload_inputs(inputs);

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("compute encoder");
        for (index, (((weight, &input_offset), &output_offset), &(rows, _))) in weights
            .iter()
            .zip(&self.input_offsets)
            .zip(&self.output_offsets)
            .zip(&self.layout)
            .enumerate()
        {
            compute::encode_gemv_q8_0_at(
                ctx,
                &encoder,
                &weight.0,
                weight.1,
                &self.input,
                input_offset,
                &self.output,
                output_offset,
                &self.n[index],
                &self.k[index],
                &self.zero_offset,
                rows,
            );
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        self.layout
            .iter()
            .zip(&self.output_offsets)
            .map(|(&(rows, _), &offset)| unsafe {
                std::slice::from_raw_parts(
                    self.output.contents().as_ptr().add(offset) as *const f32,
                    rows,
                )
                .to_vec()
            })
            .collect()
    }
}

pub(crate) struct PrefillQ8MultiCarrier {
    capacity_seq_len: usize,
    hidden_dim: usize,
    layout: Vec<(usize, usize)>,
    input: Retained<ProtocolObject<dyn MTLBuffer>>,
    input_f16: Retained<ProtocolObject<dyn MTLBuffer>>,
    outputs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    n: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    m: Retained<ProtocolObject<dyn MTLBuffer>>,
    input_elements: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl PrefillQ8MultiCarrier {
    pub(crate) fn new(
        ctx: &compute::MetalContext,
        capacity_seq_len: usize,
        layout: &[(usize, usize)],
    ) -> Self {
        assert!(!layout.is_empty());
        let hidden_dim = layout[0].1;
        assert!(layout.iter().all(|&(_, cols)| cols == hidden_dim));
        let input_elements = capacity_seq_len * hidden_dim;
        Self {
            capacity_seq_len,
            hidden_dim,
            layout: layout.to_vec(),
            input: ffn_chain::empty_f32_buf(ctx, input_elements),
            input_f16: ffn_chain::empty_f16_buf(ctx, input_elements),
            outputs: layout
                .iter()
                .map(|&(rows, _)| ffn_chain::empty_f32_buf(ctx, capacity_seq_len * rows))
                .collect(),
            n: layout
                .iter()
                .map(|&(rows, _)| ffn_chain::u32_buf(ctx, rows as u32))
                .collect(),
            k: ffn_chain::u32_buf(ctx, hidden_dim as u32),
            m: ffn_chain::u32_buf(ctx, capacity_seq_len as u32),
            input_elements: ffn_chain::u32_buf(ctx, input_elements as u32),
        }
    }

    fn store_u32(buffer: &ProtocolObject<dyn MTLBuffer>, value: u32) {
        unsafe {
            std::ptr::write(buffer.contents().as_ptr() as *mut u32, value);
        }
    }

    fn upload_input(&self, input: &[f32], seq_len: usize) {
        assert!(seq_len <= self.capacity_seq_len);
        assert_eq!(input.len(), seq_len * self.hidden_dim);
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input.contents().as_ptr() as *mut f32,
                input.len(),
            );
        }
    }

    pub(crate) fn dispatch(
        &self,
        ctx: &compute::MetalContext,
        weights: &[(Retained<ProtocolObject<dyn MTLBuffer>>, u32)],
        input: &[f32],
        seq_len: usize,
    ) -> Vec<Vec<f32>> {
        assert_eq!(weights.len(), self.layout.len());
        self.upload_input(input, seq_len);
        Self::store_u32(&self.m, seq_len as u32);
        Self::store_u32(&self.input_elements, (seq_len * self.hidden_dim) as u32);
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("compute encoder");
        compute::encode_cast_f32_to_f16(
            ctx,
            &encoder,
            &self.input,
            &self.input_f16,
            &self.input_elements,
            input.len(),
        );
        for (index, ((weight, &(rows, _)), output)) in weights
            .iter()
            .zip(&self.layout)
            .zip(&self.outputs)
            .enumerate()
        {
            compute::encode_gemm_q8_0_tensorops_v2(
                ctx,
                &encoder,
                &weight.0,
                weight.1,
                &self.input_f16,
                output,
                &self.n[index],
                &self.k,
                &self.m,
                rows,
                seq_len,
            );
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        self.layout
            .iter()
            .zip(&self.outputs)
            .map(|(&(rows, _), output)| unsafe {
                std::slice::from_raw_parts(output.contents().as_ptr() as *const f32, seq_len * rows)
                    .to_vec()
            })
            .collect()
    }
}

impl MetalBackend {
    pub fn deepseek4_q_front(
        &self,
        q_a_weight: &[u8],
        q_norm_weight: &[u8],
        output_weights: &[&[u8]],
        input: &[f32],
        hidden_dim: usize,
        q_rank: usize,
        output_layout: &[(usize, usize)],
        eps: f32,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(output_weights.len(), output_layout.len());
        self.ensure_weight_residency(ctx);
        let q_a_weight = self.glm_mla_wrap(ctx, q_a_weight);
        let q_norm_weight = self.glm_mla_wrap(ctx, q_norm_weight);
        let output_weights = output_weights
            .iter()
            .map(|&raw| self.glm_mla_wrap(ctx, raw))
            .collect::<Vec<_>>();
        let mut key = Vec::with_capacity(output_layout.len() + 1);
        key.push((q_rank, hidden_dim));
        key.extend_from_slice(output_layout);
        let mut carriers = self.deepseek4_q_front_carriers.borrow_mut();
        let carrier = carriers
            .entry(key)
            .or_insert_with(|| QFrontCarrier::new(ctx, hidden_dim, q_rank, output_layout));
        carrier.dispatch(
            ctx,
            &q_a_weight,
            &q_norm_weight,
            &output_weights,
            input,
            eps,
        )
    }

    pub fn deepseek4_q8_multi_gemv(
        &self,
        weights: &[&[u8]],
        inputs: &[&[f32]],
        layout: &[(usize, usize)],
    ) -> Vec<Vec<f32>> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        assert_eq!(weights.len(), layout.len());
        assert_eq!(inputs.len(), layout.len());
        self.ensure_weight_residency(ctx);
        let residency_enabled = self.weight_residency_enabled();
        let wrapped = weights
            .iter()
            .map(|&raw| {
                let key = resident_key(raw);
                let mut resident = self.resident.borrow_mut();
                let entry = resident
                    .entry(key)
                    .or_insert_with(|| resident_cache_entry(ctx, raw));
                if residency_enabled {
                    if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                        lru.touch(key, &entry.0);
                    }
                }
                (entry.0.clone(), entry.1 as usize)
            })
            .collect::<Vec<_>>();
        if residency_enabled {
            if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                lru.commit_if_dirty();
            }
        }
        let key = layout.to_vec();
        let mut carriers = self.deepseek4_q8_multi_carriers.borrow_mut();
        let carrier = carriers
            .entry(key)
            .or_insert_with(|| Q8MultiGemvCarrier::new(ctx, layout));
        carrier.dispatch(ctx, &wrapped, inputs)
    }

    pub fn deepseek4_prefill_q8_multi_gemm(
        &self,
        weights: &[&[u8]],
        input: &[f32],
        seq_len: usize,
        layout: &[(usize, usize)],
    ) -> Option<Vec<Vec<f32>>> {
        let ctx = self.ctx.as_ref().expect("MetalBackend: no Metal context");
        if !ctx.tensorops_capable
            || ctx.cast_f32_f16_pipeline.is_none()
            || ctx.gemm_q8_0_tensorops_v2_pipeline.is_none()
            || std::env::var("RNB_METAL_PREFILL_GDN_PROJ_V2").as_deref() == Ok("0")
        {
            return None;
        }
        assert_eq!(weights.len(), layout.len());
        assert_eq!(input.len(), seq_len * layout[0].1);
        self.ensure_weight_residency(ctx);
        let residency_enabled = self.weight_residency_enabled();
        let wrapped = weights
            .iter()
            .map(|&raw| {
                let key = resident_key(raw);
                let mut resident = self.resident.borrow_mut();
                let entry = resident
                    .entry(key)
                    .or_insert_with(|| resident_cache_entry(ctx, raw));
                if residency_enabled {
                    if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                        lru.touch(key, &entry.0);
                    }
                }
                (entry.0.clone(), entry.1)
            })
            .collect::<Vec<_>>();
        if residency_enabled {
            if let Some(lru) = self.weight_residency.borrow_mut().as_mut() {
                lru.commit_if_dirty();
            }
        }
        let mut carriers = self.deepseek4_prefill_q8_multi_carriers.borrow_mut();
        let carrier = carriers
            .entry(layout.to_vec())
            .or_insert_with(|| PrefillQ8MultiCarrier::new(ctx, seq_len, layout));
        if carrier.capacity_seq_len < seq_len {
            *carrier = PrefillQ8MultiCarrier::new(ctx, seq_len, layout);
        }
        Some(carrier.dispatch(ctx, &wrapped, input, seq_len))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn q_front_keeps_q5_rms_q8_intermediate_on_device() {
        let backend = MetalBackend::new();
        if backend.ctx.is_none() {
            return;
        }

        let hidden_dim = 256;
        let q_rank = 32;
        let output_rows = 37;
        let q_a_raw = crate::tests_fixture::scaled_q5k_matrix(q_rank, hidden_dim, 13);
        let q_b_raw = crate::tests_fixture::scaled_q8_0_matrix(output_rows, q_rank, 29);
        let q_norm = (0..q_rank)
            .map(|index| 0.75 + index as f32 * 0.01)
            .collect::<Vec<_>>();
        let input = (0..hidden_dim)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        let q_a_tensor = rnb_core::tensor::Tensor::from_slice(&q_a_raw, &[q_a_raw.len()]);
        let q_b_tensor = rnb_core::tensor::Tensor::from_slice(&q_b_raw, &[q_b_raw.len()]);
        let q_norm_tensor = rnb_core::tensor::Tensor::from_slice(&q_norm, &[q_norm.len()]);
        q_a_tensor.register_host_storage();
        q_b_tensor.register_host_storage();
        q_norm_tensor.register_host_storage();

        let (actual_norm, actual_outputs) = backend.deepseek4_q_front(
            q_a_tensor.as_bytes().expect("Q5_K bytes"),
            q_norm_tensor.as_bytes().expect("RMS weight bytes"),
            &[q_b_tensor.as_bytes().expect("Q8_0 bytes")],
            &input,
            hidden_dim,
            q_rank,
            &[(output_rows, q_rank)],
            1.0e-6,
        );

        let mut expected_q_a = Vec::with_capacity(q_rank);
        for row in q_a_raw.chunks_exact(176) {
            let dequant = crate::tests_fixture::q5k_dequant(row);
            expected_q_a.push(
                dequant
                    .iter()
                    .zip(&input)
                    .map(|(&weight, &value)| weight * value)
                    .sum::<f32>(),
            );
        }
        let scale = (expected_q_a.iter().map(|value| value * value).sum::<f32>() / q_rank as f32
            + 1.0e-6)
            .sqrt()
            .recip();
        let expected_norm = expected_q_a
            .iter()
            .zip(&q_norm)
            .map(|(&value, &gain)| value * scale * gain)
            .collect::<Vec<_>>();
        let expected_output = q_b_raw
            .chunks_exact(34)
            .map(|row| {
                crate::tests_fixture::q8_0_dequant(row)
                    .iter()
                    .zip(&expected_norm)
                    .map(|(&weight, &value)| weight * value)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();

        let max_norm_error = actual_norm
            .iter()
            .zip(&expected_norm)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let max_output_error = actual_outputs[0]
            .iter()
            .zip(&expected_output)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_norm_error < 1.0e-4, "max norm error {max_norm_error}");
        assert!(
            max_output_error < 1.0e-4,
            "max output error {max_output_error}"
        );
    }

    #[test]
    fn prefill_carrier_cache_is_layout_bounded_and_grows_in_place() {
        let backend = MetalBackend::new();
        let Some(ctx) = backend.ctx.as_ref() else {
            return;
        };
        if !ctx.tensorops_capable
            || ctx.cast_f32_f16_pipeline.is_none()
            || ctx.gemm_q8_0_tensorops_v2_pipeline.is_none()
            || std::env::var("RNB_METAL_PREFILL_GDN_PROJ_V2").as_deref() == Ok("0")
        {
            return;
        }

        let layout = [(64, 32)];
        let weight = crate::tests_fixture::scaled_q8_0_matrix(64, 32, 17);
        let weight = rnb_core::tensor::Tensor::from_slice(&weight, &[weight.len()]);
        weight.register_host_storage();
        let weights = [weight.as_bytes().expect("Q8_0 weight bytes")];
        for seq_len in [2, 4, 1] {
            let input = vec![0.25f32; seq_len * 32];
            let output = backend
                .deepseek4_prefill_q8_multi_gemm(&weights, &input, seq_len, &layout)
                .expect("DeepSeek4 prefill Q8_0 projection");
            assert_eq!(output[0].len(), seq_len * 64);
        }

        let carriers = backend.deepseek4_prefill_q8_multi_carriers.borrow();
        assert_eq!(carriers.len(), 1);
        assert_eq!(carriers[layout.as_slice()].capacity_seq_len, 4);
    }
}
