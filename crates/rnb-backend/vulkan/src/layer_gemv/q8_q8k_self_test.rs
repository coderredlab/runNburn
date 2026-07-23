use super::*;

impl VulkanLayerGemv {
    /// Verify Q8_0 weights × Q8K activations against an integer CPU reference
    /// on the real GPU.
    pub fn self_test_q8_q8k(&mut self) -> Result<f32, String> {
        self.self_test_quantize_to_q8k()?;
        unsafe {
            self.ensure_pipeline_q8_q8k()?;
        }

        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.41).sin() * 1.7) - 0.3 + ((i as f32 * 0.13).cos() * 0.9);
        }
        let amax = input.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let d_act = amax / 127.0;
        let inv_d_act = if d_act != 0.0 { 1.0 / d_act } else { 0.0 };
        let mut q8k = [0i8; 256];
        for (q, &value) in q8k.iter_mut().zip(&input) {
            *q = (value * inv_d_act).round().clamp(-128.0, 127.0) as i8;
        }

        // Eight Q8_0 blocks cover one 256-element Q8K activation block.
        let mut weights = vec![0u8; 8 * 34];
        let mut cpu_dot = 0.0f32;
        for block_index in 0..8usize {
            let block = &mut weights[block_index * 34..(block_index + 1) * 34];
            let d_weight = 0.0075 + block_index as f32 * 0.00125;
            block[0..2].copy_from_slice(&half::f16::from_f32(d_weight).to_le_bytes());
            let stored_scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let mut isum = 0i32;
            for element in 0..32usize {
                let value = (((block_index * 47 + element * 29 + 11) % 255) as i32 - 127) as i8;
                block[2 + element] = value as u8;
                isum += value as i32 * q8k[block_index * 32 + element] as i32;
            }
            cpu_dot += stored_scale * d_act * isum as f32;
        }

        let test_id = crate::weight_cache::WeightId {
            layer: 252,
            kind: crate::weight_cache::WeightKind::QProj,
        };

        unsafe {
            let gpu_buf = self.cache.get_or_upload(
                &self.ctx,
                self.command_pool,
                test_id,
                &weights,
                1,
                256,
                QuantType::Q8_0,
                self.mode,
            )?;
            let weight_handle = gpu_buf.buffer;
            let weight_size = gpu_buf.size;
            let q8k_buf = self.q8k_input_buf.as_ref().unwrap();
            let output_buf = self.output_bufs[0].as_ref().unwrap();
            let dot_pipe = self.pipeline_q8_q8k.as_ref().unwrap();
            dot_pipe.bind_buffers_with_offsets(
                &self.ctx,
                0,
                &GpuBuffer {
                    buffer: weight_handle,
                    memory: 0,
                    size: weight_size,
                },
                0,
                weight_size,
                q8k_buf,
                0,
                69 * 4,
                output_buf,
                0,
                4,
            );

            (self.ctx.vk.reset_command_buffer)(self.command_buffer, 0);
            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                p_inheritance_info: ptr::null(),
            };
            (self.ctx.vk.begin_command_buffer)(self.command_buffer, &begin_info);
            (self.ctx.vk.cmd_bind_pipeline)(
                self.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                dot_pipe.pipeline,
            );
            (self.ctx.vk.cmd_bind_descriptor_sets)(
                self.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                dot_pipe.pipeline_layout,
                0,
                1,
                &dot_pipe.descriptor_sets[0],
                0,
                ptr::null(),
            );
            #[repr(C)]
            struct PC {
                rows: u32,
                cols: u32,
                rows_per_wg: u32,
            }
            let push = PC {
                rows: 1,
                cols: 256,
                rows_per_wg: LOCAL_SIZE_X,
            };
            (self.ctx.vk.cmd_push_constants)(
                self.command_buffer,
                dot_pipe.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                12,
                &push as *const _ as *const std::ffi::c_void,
            );
            (self.ctx.vk.cmd_dispatch)(self.command_buffer, 1, 1, 1);

            let barrier = VkBufferMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_HOST_READ_BIT,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                buffer: output_buf.buffer,
                offset: 0,
                size: VK_WHOLE_SIZE,
            };
            (self.ctx.vk.cmd_pipeline_barrier)(
                self.command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0,
                0,
                ptr::null(),
                1,
                &barrier,
                0,
                ptr::null(),
            );
            (self.ctx.vk.end_command_buffer)(self.command_buffer);
            (self.ctx.vk.reset_fences)(self.ctx.device, 1, &self.fence);
            let submit_info = VkSubmitInfo {
                s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffer,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            };
            (self.ctx.vk.queue_submit)(self.ctx.queue, 1, &submit_info, self.fence);
            (self.ctx.vk.wait_for_fences)(self.ctx.device, 1, &self.fence, 1, 10_000_000_000u64);

            let mut gpu_dot = 0.0f32;
            ptr::copy_nonoverlapping(
                self.output_mapped_ptrs[0],
                &mut gpu_dot as *mut f32 as *mut u8,
                4,
            );
            let diff = (cpu_dot - gpu_dot).abs();
            eprintln!("[q8_q8k self_test] CPU={cpu_dot:.6} GPU={gpu_dot:.6} diff={diff:.6}");
            Ok(diff)
        }
    }
}
