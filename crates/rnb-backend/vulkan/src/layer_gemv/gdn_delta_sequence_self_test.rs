use super::*;

impl VulkanLayerGemv {
    /// Verify the 128×128 parallel GDN recurrence against the scalar CPU contract.
    pub fn self_test_gdn_delta_sequence_d128(&mut self) -> Result<f32, String> {
        const SEQ_LEN: usize = 3;
        const NUM_HEADS: usize = 2;
        const NUM_K_HEADS: usize = 1;
        const HEAD_DIM: usize = 128;
        const CONV_CHANNELS: usize = (2 * NUM_K_HEADS + NUM_HEADS) * HEAD_DIM;
        const PARAM_STRIDE: usize = 2 * NUM_K_HEADS + 2 * NUM_HEADS;
        const STATE_PER_HEAD: usize = HEAD_DIM * HEAD_DIM;

        let mut conv = vec![0.0f32; SEQ_LEN * CONV_CHANNELS];
        for (i, value) in conv.iter_mut().enumerate() {
            *value = ((i * 17 + 5) % 41) as f32 * 0.004 - 0.08;
        }
        let mut params = vec![0.0f32; SEQ_LEN * PARAM_STRIDE];
        for token in 0..SEQ_LEN {
            let base = token * PARAM_STRIDE;
            for key_head in 0..NUM_K_HEADS {
                params[base + key_head] = 0.8 + key_head as f32 * 0.05;
                params[base + NUM_K_HEADS + key_head] = 0.7;
            }
            for head in 0..NUM_HEADS {
                params[base + 2 * NUM_K_HEADS + head] = 0.35 + token as f32 * 0.03;
                params[base + 2 * NUM_K_HEADS + NUM_HEADS + head] = 0.91 + head as f32 * 0.02;
            }
        }
        let mut initial_state = vec![0.0f32; NUM_HEADS * STATE_PER_HEAD];
        for (i, value) in initial_state.iter_mut().enumerate() {
            *value = ((i * 13 + 3) % 29) as f32 * 0.0005 - 0.007;
        }

        let mut expected_state = initial_state.clone();
        let mut expected_output = vec![0.0f32; SEQ_LEN * NUM_HEADS * HEAD_DIM];
        for token in 0..SEQ_LEN {
            let conv_base = token * CONV_CHANNELS;
            let param_base = token * PARAM_STRIDE;
            for head in 0..NUM_HEADS {
                let key_head = head % NUM_K_HEADS;
                let q_base = conv_base + key_head * HEAD_DIM;
                let k_base = conv_base + (NUM_K_HEADS + key_head) * HEAD_DIM;
                let v_base = conv_base + (2 * NUM_K_HEADS + head) * HEAD_DIM;
                let state_base = head * STATE_PER_HEAD;
                let q_inv = params[param_base + key_head];
                let k_inv = params[param_base + NUM_K_HEADS + key_head];
                let beta = params[param_base + 2 * NUM_K_HEADS + head];
                let decay = params[param_base + 2 * NUM_K_HEADS + NUM_HEADS + head];

                for vi in 0..HEAD_DIM {
                    let row_base = state_base + vi * HEAD_DIM;
                    let mut state_dot_k = 0.0f32;
                    for ki in 0..HEAD_DIM {
                        state_dot_k +=
                            decay * expected_state[row_base + ki] * conv[k_base + ki] * k_inv;
                    }
                    let delta = (conv[v_base + vi] - state_dot_k) * beta;
                    let mut output = 0.0f32;
                    for ki in 0..HEAD_DIM {
                        let state_index = row_base + ki;
                        let updated =
                            decay * expected_state[state_index] + conv[k_base + ki] * k_inv * delta;
                        expected_state[state_index] = updated;
                        output += conv[q_base + ki] * q_inv * updated;
                    }
                    expected_output[(token * NUM_HEADS + head) * HEAD_DIM + vi] = output;
                }
            }
        }

        let mut buffers = Vec::with_capacity(7);
        let result = (|| unsafe {
            let usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            let memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            for bytes in [
                conv.len() * 4,
                params.len() * 4,
                4,
                4,
                4,
                initial_state.len() * 4,
                expected_output.len() * 4,
            ] {
                buffers.push(self.ctx.create_buffer(bytes as u64, usage, memory)?);
            }
            self.ctx
                .upload_to_buffer(&buffers[0], bytemuck_cast(&conv))?;
            self.ctx
                .upload_to_buffer(&buffers[1], bytemuck_cast(&params))?;
            self.ctx.upload_to_buffer(&buffers[2], &[0; 4])?;
            self.ctx.upload_to_buffer(&buffers[3], &[0; 4])?;
            self.ctx.upload_to_buffer(&buffers[4], &[0; 4])?;
            self.ctx
                .upload_to_buffer(&buffers[5], bytemuck_cast(&initial_state))?;
            self.ctx.upload_to_buffer(
                &buffers[6],
                bytemuck_cast(&vec![0.0f32; expected_output.len()]),
            )?;

            let cmdbuf = self.begin_recording()?;
            let pipe = self
                .pipeline_gdn_delta_sequence_d128
                .as_ref()
                .ok_or("gdn delta sequence d128 pipeline not initialized")?;
            pipe.bind_n_buffers_with_offsets(
                &self.ctx,
                0,
                &[
                    (&buffers[0], 0, buffers[0].size),
                    (&buffers[1], 0, buffers[1].size),
                    (&buffers[2], 0, buffers[2].size),
                    (&buffers[3], 0, buffers[3].size),
                    (&buffers[4], 0, buffers[4].size),
                    (&buffers[5], 0, buffers[5].size),
                    (&buffers[6], 0, buffers[6].size),
                ],
            );

            (self.ctx.vk.cmd_bind_pipeline)(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
            (self.ctx.vk.cmd_bind_descriptor_sets)(
                cmdbuf,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipe.pipeline_layout,
                0,
                1,
                &pipe.descriptor_sets[0],
                0,
                ptr::null(),
            );
            let push = [
                CONV_CHANNELS as u32,
                (NUM_HEADS * HEAD_DIM) as u32,
                NUM_K_HEADS as u32,
                NUM_HEADS as u32,
                HEAD_DIM as u32,
                HEAD_DIM as u32,
                0,
                SEQ_LEN as u32,
                CONV_CHANNELS as u32,
                HEAD_DIM as u32,
                (NUM_HEADS * HEAD_DIM) as u32,
            ];
            (self.ctx.vk.cmd_push_constants)(
                cmdbuf,
                pipe.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                44,
                push.as_ptr() as *const std::ffi::c_void,
            );
            (self.ctx.vk.cmd_dispatch)(cmdbuf, NUM_HEADS as u32, 32, 1);

            let barriers = [
                VkBufferMemoryBarrier {
                    s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                    dst_access_mask: VK_ACCESS_HOST_READ_BIT,
                    src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                    buffer: buffers[5].buffer,
                    offset: 0,
                    size: VK_WHOLE_SIZE,
                },
                VkBufferMemoryBarrier {
                    s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                    dst_access_mask: VK_ACCESS_HOST_READ_BIT,
                    src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                    buffer: buffers[6].buffer,
                    offset: 0,
                    size: VK_WHOLE_SIZE,
                },
            ];
            (self.ctx.vk.cmd_pipeline_barrier)(
                cmdbuf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0,
                0,
                ptr::null(),
                barriers.len() as u32,
                barriers.as_ptr(),
                0,
                ptr::null(),
            );
            self.submit_and_wait()?;

            let mut gpu_state = vec![0.0f32; expected_state.len()];
            let mut gpu_output = vec![0.0f32; expected_output.len()];
            self.ctx
                .download_from_buffer(&buffers[5], bytemuck_cast_mut(&mut gpu_state))?;
            self.ctx
                .download_from_buffer(&buffers[6], bytemuck_cast_mut(&mut gpu_output))?;
            let state_diff = expected_state
                .iter()
                .zip(&gpu_state)
                .map(|(cpu, gpu)| (cpu - gpu).abs())
                .fold(0.0f32, f32::max);
            let output_diff = expected_output
                .iter()
                .zip(&gpu_output)
                .map(|(cpu, gpu)| (cpu - gpu).abs())
                .fold(0.0f32, f32::max);
            let state_max = expected_state
                .iter()
                .zip(&gpu_state)
                .enumerate()
                .max_by(|(_, (cpu_a, gpu_a)), (_, (cpu_b, gpu_b))| {
                    (**cpu_a - **gpu_a)
                        .abs()
                        .total_cmp(&(**cpu_b - **gpu_b).abs())
                })
                .unwrap();
            let output_max = expected_output
                .iter()
                .zip(&gpu_output)
                .enumerate()
                .max_by(|(_, (cpu_a, gpu_a)), (_, (cpu_b, gpu_b))| {
                    (**cpu_a - **gpu_a)
                        .abs()
                        .total_cmp(&(**cpu_b - **gpu_b).abs())
                })
                .unwrap();
            eprintln!(
                "[gdn_delta_sequence_d128 self_test] state_max={state_max:?} output_max={output_max:?}"
            );
            let max_diff = state_diff.max(output_diff);
            eprintln!(
                "[gdn_delta_sequence_d128 self_test] state_diff={state_diff:.6} output_diff={output_diff:.6}"
            );
            Ok(max_diff)
        })();

        unsafe {
            for buffer in buffers {
                self.ctx.destroy_buffer(buffer);
            }
        }
        result
    }
}
