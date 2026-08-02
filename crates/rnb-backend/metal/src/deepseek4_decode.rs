use super::*;

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
