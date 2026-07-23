use super::*;

impl VulkanLayerGemv {
    /// Verify the Q5_K weight × Q8K activation integer-dot path on the real GPU.
    pub fn self_test_q5k_q8k(&mut self) -> Result<f32, String> {
        unsafe {
            self.ensure_pipeline_quantize_to_q8k()?;
            self.ensure_pipeline_q5k_q8k()?;
            self.ensure_q8k_input_buf(1)?;
        }

        // Q5_K block (176 bytes): d, dmin, scales/mins, qh, qs.
        let mut block = [0u8; 176];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
        let scale_bytes: [u8; 12] = [
            0b00_110100,
            0b01_010110,
            0b10_101010,
            0b11_001111,
            0b01_011001,
            0b10_100011,
            0b00_010101,
            0b11_111000,
            0b1010_0111,
            0b0011_1100,
            0b1110_0010,
            0b0101_1001,
        ];
        block[4..16].copy_from_slice(&scale_bytes);
        for i in 0..32 {
            block[16 + i] = ((i * 37 + 13) % 256) as u8;
        }
        for i in 0..128 {
            block[48 + i] = ((i * 7 + 11) % 256) as u8;
        }

        // Deterministic input with mixed signs (negative components must
        // round-trip correctly through the Adreno shift workaround).
        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.41).sin() * 1.7) - 0.3 + ((i as f32 * 0.13).cos() * 0.9);
        }

        // CPU reference: replicate `dot_q4_k_q8k_scalar` semantics.
        let mut amax = 0.0f32;
        for &x in input.iter() {
            amax = amax.max(x.abs());
        }
        let d_act = amax / 127.0;
        let id = if d_act != 0.0 { 1.0 / d_act } else { 0.0 };
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 8];
        for i in 0..256 {
            let q = (input[i] * id).round().clamp(-128.0, 127.0) as i8;
            qs[i] = q;
            bsums[i / 32] += q as i16;
        }

        let d_w = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..8usize {
            if j < 4 {
                sc[j] = block[4 + j] & 0x3F;
                mn[j] = block[4 + j + 4] & 0x3F;
            } else {
                sc[j] = (block[4 + j + 4] & 0x0F) | ((block[4 + j - 4] >> 6) << 4);
                mn[j] = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
            }
        }
        let qh = &block[16..48];
        let qs_w = &block[48..176];
        let mut sumi = 0i32;
        let mut summ = 0i32;
        for group in 0..4 {
            let q_off = group * 32;
            let x_off = group * 64;
            let is = group * 2;
            let mut isum0 = 0i32;
            let mut isum1 = 0i32;
            for l in 0..32 {
                let high_lo = if qh[l] & (1 << (group * 2)) != 0 {
                    16
                } else {
                    0
                };
                let high_hi = if qh[l] & (2 << (group * 2)) != 0 {
                    16
                } else {
                    0
                };
                let lo = (qs_w[q_off + l] & 0x0F) + high_lo;
                let hi = (qs_w[q_off + l] >> 4) + high_hi;
                isum0 += lo as i32 * qs[x_off + l] as i32;
                isum1 += hi as i32 * qs[x_off + 32 + l] as i32;
            }
            sumi += sc[is] as i32 * isum0 + sc[is + 1] as i32 * isum1;
            summ += mn[is] as i32 * bsums[group * 2] as i32
                + mn[is + 1] as i32 * bsums[group * 2 + 1] as i32;
        }
        let cpu_dot = d_act * (d_w * sumi as f32 - dmin * summ as f32);

        // Repack weight to transposed SoA (1 row, 1 block) and upload.
        let repacked = crate::gemv::repack_q5k_transposed(&block, 1, 1);

        let test_id = crate::weight_cache::WeightId {
            layer: 251,
            kind: crate::weight_cache::WeightKind::QProj,
        };

        let input_bytes = input.len() * std::mem::size_of::<f32>();

        unsafe {
            ptr::copy_nonoverlapping(
                input.as_ptr() as *const u8,
                self.input_mapped_ptr,
                input_bytes,
            );

            // `repack_q5k_transposed` already produced SoA, but `cache::get_or_upload`
            // re-repacks AoS→SoA based on quant. Pass the original 176-byte AoS
            // block so the cache does the canonical repack and we don't ship
            // double-packed bytes.
            let aos_bytes: &[u8] = &block;
            let _ = repacked;
            let gpu_buf = self.cache.get_or_upload(
                &self.ctx,
                self.command_pool,
                test_id,
                aos_bytes,
                1,
                256,
                QuantType::Q5K,
                self.mode,
            )?;
            let weight_handle = gpu_buf.buffer;
            let weight_size = gpu_buf.size;

            let q8k_buf = self.q8k_input_buf.as_ref().unwrap();
            let q8k_total: u64 = 69 * 4;
            let output_buf = self.output_bufs[0].as_ref().unwrap();

            let q_pipe = self.pipeline_quantize_to_q8k.as_ref().unwrap();
            q_pipe.bind_n_buffers_with_offsets(
                &self.ctx,
                0,
                &[
                    (self.input_buf.as_ref().unwrap(), 0, input_bytes as u64),
                    (q8k_buf, 0, q8k_total),
                ],
            );

            let dot_pipe = self.pipeline_q5k_q8k.as_ref().unwrap();
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
                q8k_total,
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

            // Pass 1: quantize_to_q8k.
            (self.ctx.vk.cmd_bind_pipeline)(
                self.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                q_pipe.pipeline,
            );
            (self.ctx.vk.cmd_bind_descriptor_sets)(
                self.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                q_pipe.pipeline_layout,
                0,
                1,
                &q_pipe.descriptor_sets[0],
                0,
                ptr::null(),
            );
            let nb_push: u32 = 1;
            (self.ctx.vk.cmd_push_constants)(
                self.command_buffer,
                q_pipe.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                4,
                &nb_push as *const _ as *const std::ffi::c_void,
            );
            (self.ctx.vk.cmd_dispatch)(self.command_buffer, 1, 1, 1);

            // Barrier between quantize write and dot read.
            let q_barrier = VkBufferMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                buffer: q8k_buf.buffer,
                offset: 0,
                size: VK_WHOLE_SIZE,
            };
            (self.ctx.vk.cmd_pipeline_barrier)(
                self.command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                ptr::null(),
                1,
                &q_barrier,
                0,
                ptr::null(),
            );

            // Pass 2: q5k_q8k.
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

            let out_barrier = VkBufferMemoryBarrier {
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
                &out_barrier,
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
            eprintln!("[q5k_q8k self_test] CPU={cpu_dot:.6} GPU={gpu_dot:.6} diff={diff:.6}");
            Ok(diff)
        }
    }
}
