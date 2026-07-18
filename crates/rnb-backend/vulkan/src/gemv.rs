use crate::context::{GpuBuffer, VulkanContext};
use crate::ffi::types::*;
use crate::pipeline::ComputePipeline;
use crate::spirv::builder::emit_q8_0_gemv;
use std::ptr;

const LOCAL_SIZE_X: u32 = 64;

/// Push constants layout matching the SPIR-V shader.
#[repr(C)]
struct PushConstants {
    rows: u32,
    cols: u32,
    rows_per_wg: u32,
}

/// Public API: Vulkan-accelerated Q8_0 GEMV.
///
/// Owns: VulkanContext, ComputePipeline, weight/input/output GpuBuffers.
/// Weight buffer is device-local. Input/output buffers are host-visible with persistent mapping.
/// Command buffer is pre-recorded once at init — gemv() just does memcpy + submit + memcpy.
pub struct VulkanGemv {
    ctx: VulkanContext,
    pipeline: Option<ComputePipeline>,
    weight_buf: Option<GpuBuffer>,
    input_buf: Option<GpuBuffer>,
    output_buf: Option<GpuBuffer>,
    input_mapped_ptr: *mut u8,
    output_mapped_ptr: *const u8,
    rows: u32,
    cols: u32,
    command_pool: VkCommandPool,
    command_buffer: VkCommandBuffer,
    fence: VkFence,
}

unsafe impl Send for VulkanGemv {}
unsafe impl Sync for VulkanGemv {}

impl VulkanGemv {
    /// Initialize Vulkan, create SPIR-V shader, pipeline, buffers, upload weights.
    ///
    /// `weight_bytes`: raw Q8_0-quantized weight data (rows x cols bytes, each row
    ///                 is packed as Q8_0 blocks of 34 bytes each).
    /// `rows`: number of output rows.
    /// `cols`: number of input columns (must be a multiple of 32).
    pub fn new(weight_bytes: &[u8], rows: usize, cols: usize) -> Result<Self, String> {
        unsafe {
            let ctx = VulkanContext::new()?;

            // Generate SPIR-V
            let spirv = emit_q8_0_gemv(LOCAL_SIZE_X);

            // Create compute pipeline (1 descriptor set for single dispatch)
            let pipeline = ComputePipeline::new(&ctx, &spirv, 1)?;

            // Create command pool, command buffer, fence
            let cmd_pool_create_info = VkCommandPoolCreateInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0x00000002, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
                queue_family_index: ctx.queue_family_index,
            };
            let mut command_pool: VkCommandPool = VK_NULL_HANDLE;
            let res = (ctx.vk.create_command_pool)(
                ctx.device,
                &cmd_pool_create_info,
                ptr::null(),
                &mut command_pool,
            );
            if res != VK_SUCCESS {
                pipeline.destroy(&ctx);
                return Err(format!("vkCreateCommandPool failed: {}", res));
            }
            let cb_alloc_info = VkCommandBufferAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: ptr::null(),
                command_pool,
                level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                command_buffer_count: 1,
            };
            let mut command_buffer: VkCommandBuffer = ptr::null_mut();
            let res =
                (ctx.vk.allocate_command_buffers)(ctx.device, &cb_alloc_info, &mut command_buffer);
            if res != VK_SUCCESS {
                (ctx.vk.destroy_command_pool)(ctx.device, command_pool, ptr::null());
                pipeline.destroy(&ctx);
                return Err(format!("vkAllocateCommandBuffers failed: {}", res));
            }
            let fence_create_info = VkFenceCreateInfo {
                s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: VK_FENCE_CREATE_SIGNALED_BIT,
            };
            let mut fence: VkFence = VK_NULL_HANDLE;
            let res =
                (ctx.vk.create_fence)(ctx.device, &fence_create_info, ptr::null(), &mut fence);
            if res != VK_SUCCESS {
                (ctx.vk.destroy_command_pool)(ctx.device, command_pool, ptr::null());
                pipeline.destroy(&ctx);
                return Err(format!("vkCreateFence failed: {}", res));
            }

            // Repack Q8_0 weights to transposed SoA layout for coalesced GPU access
            let blocks_per_row = cols / 32;
            let repacked = repack_q8_0_transposed(weight_bytes, rows, blocks_per_row);

            // Buffer sizes (repacked is Vec<u32>)
            let weight_size = (repacked.len() * 4) as u64;
            let input_size = (cols * std::mem::size_of::<f32>()) as u64;
            let output_size = (rows * std::mem::size_of::<f32>()) as u64;

            let host_visible =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            // --- Step 1: Device-local weight buffer with staging upload ---
            // Convert repacked u32 data to byte slice for upload
            let repacked_bytes =
                std::slice::from_raw_parts(repacked.as_ptr() as *const u8, repacked.len() * 4);

            let device_local_usage =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            let weight_buf = match ctx.create_buffer(
                weight_size,
                device_local_usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            ) {
                Ok(buf) => {
                    // Successfully allocated device-local — upload via staging buffer
                    let staging_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    let staging_buf =
                        ctx.create_buffer(weight_size, staging_usage, host_visible)?;
                    ctx.upload_to_buffer(&staging_buf, repacked_bytes)?;
                    ctx.copy_buffer_and_wait(command_pool, &staging_buf, &buf, weight_size)?;
                    ctx.destroy_buffer(staging_buf);
                    buf
                }
                Err(_) => {
                    // Fallback: host-visible weight buffer
                    eprintln!("[vulkan] device-local alloc failed, falling back to host-visible weight buffer");
                    let buf = ctx.create_buffer(
                        weight_size,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        host_visible,
                    )?;
                    ctx.upload_to_buffer(&buf, repacked_bytes)?;
                    buf
                }
            };

            // --- Step 2: Host-visible input/output buffers with persistent map ---
            let input_buf = match ctx.create_buffer(
                input_size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                host_visible,
            ) {
                Ok(b) => b,
                Err(e) => {
                    ctx.destroy_buffer(weight_buf);
                    return Err(e);
                }
            };
            let output_buf = match ctx.create_buffer(
                output_size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                host_visible,
            ) {
                Ok(b) => b,
                Err(e) => {
                    ctx.destroy_buffer(input_buf);
                    ctx.destroy_buffer(weight_buf);
                    return Err(e);
                }
            };

            let input_mapped_ptr = match ctx.map_buffer_persistent(&input_buf) {
                Ok(p) => p,
                Err(e) => {
                    ctx.destroy_buffer(output_buf);
                    ctx.destroy_buffer(input_buf);
                    ctx.destroy_buffer(weight_buf);
                    return Err(e);
                }
            };
            let output_mapped_ptr = match ctx.map_buffer_persistent(&output_buf) {
                Ok(p) => p as *const u8,
                Err(e) => {
                    ctx.unmap_buffer(&input_buf);
                    ctx.destroy_buffer(output_buf);
                    ctx.destroy_buffer(input_buf);
                    ctx.destroy_buffer(weight_buf);
                    return Err(e);
                }
            };

            // --- Step 3: Bind descriptors ---
            pipeline.bind_buffers(
                &ctx,
                0,
                &weight_buf,
                weight_size,
                &input_buf,
                input_size,
                &output_buf,
                output_size,
            );

            // --- Step 4: Pre-record command buffer ---
            let rows_u32 = rows as u32;
            let cols_u32 = cols as u32;

            // Reset fence before first use (it starts signaled)
            let res = (ctx.vk.reset_fences)(ctx.device, 1, &fence);
            if res != VK_SUCCESS {
                return Err(format!("vkResetFences (init) failed: {}", res));
            }

            // Begin command buffer with SIMULTANEOUS_USE_BIT so we can resubmit
            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
                p_inheritance_info: ptr::null(),
            };
            let res = (ctx.vk.begin_command_buffer)(command_buffer, &begin_info);
            if res != VK_SUCCESS {
                return Err(format!("vkBeginCommandBuffer (pre-record) failed: {}", res));
            }

            // Bind pipeline
            (ctx.vk.cmd_bind_pipeline)(
                command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline,
            );

            // Bind descriptor sets
            (ctx.vk.cmd_bind_descriptor_sets)(
                command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline_layout,
                0,
                1,
                &pipeline.descriptor_sets[0],
                0,
                ptr::null(),
            );

            // Push constants
            let push = PushConstants {
                rows: rows_u32,
                cols: cols_u32,
                rows_per_wg: LOCAL_SIZE_X,
            };
            (ctx.vk.cmd_push_constants)(
                command_buffer,
                pipeline.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                std::mem::size_of::<PushConstants>() as u32,
                &push as *const PushConstants as *const std::ffi::c_void,
            );

            // Dispatch
            let num_wg = (rows_u32 + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
            (ctx.vk.cmd_dispatch)(command_buffer, num_wg, 1, 1);

            // Pipeline barrier: compute shader write -> host read
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
            (ctx.vk.cmd_pipeline_barrier)(
                command_buffer,
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

            // End recording
            let res = (ctx.vk.end_command_buffer)(command_buffer);
            if res != VK_SUCCESS {
                return Err(format!("vkEndCommandBuffer (pre-record) failed: {}", res));
            }

            eprintln!(
                "[vulkan] GPU GEMV initialized: {}x{} (device-local weight, persistent map, pre-recorded cmd)",
                rows, cols
            );

            Ok(Self {
                ctx,
                pipeline: Some(pipeline),
                weight_buf: Some(weight_buf),
                input_buf: Some(input_buf),
                output_buf: Some(output_buf),
                input_mapped_ptr,
                output_mapped_ptr,
                rows: rows_u32,
                cols: cols_u32,
                command_pool,
                command_buffer,
                fence,
            })
        }
    }

    /// Run one GEMV: memcpy input, submit pre-recorded cmd, wait, memcpy output.
    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if input.len() != self.cols as usize {
            return Err(format!("input len {} != cols {}", input.len(), self.cols));
        }
        if output.len() != self.rows as usize {
            return Err(format!("output len {} != rows {}", output.len(), self.rows));
        }

        unsafe {
            // Upload input via persistent mapped pointer
            let input_bytes = input.len() * std::mem::size_of::<f32>();
            ptr::copy_nonoverlapping(
                input.as_ptr() as *const u8,
                self.input_mapped_ptr,
                input_bytes,
            );

            // Reset fence
            let res = (self.ctx.vk.reset_fences)(self.ctx.device, 1, &self.fence);
            if res != VK_SUCCESS {
                return Err(format!("vkResetFences failed: {}", res));
            }

            // Submit pre-recorded command buffer (NO re-recording)
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
            let res = (self.ctx.vk.queue_submit)(self.ctx.queue, 1, &submit_info, self.fence);
            if res != VK_SUCCESS {
                return Err(format!("vkQueueSubmit failed: {}", res));
            }

            // Wait for completion (timeout: 10 seconds)
            let res = (self.ctx.vk.wait_for_fences)(
                self.ctx.device,
                1,
                &self.fence,
                1u32,
                10_000_000_000u64,
            );
            if res != VK_SUCCESS {
                return Err(format!("vkWaitForFences failed: {}", res));
            }

            // Download output via persistent mapped pointer
            let output_bytes = output.len() * std::mem::size_of::<f32>();
            ptr::copy_nonoverlapping(
                self.output_mapped_ptr,
                output.as_mut_ptr() as *mut u8,
                output_bytes,
            );

            Ok(())
        }
    }
}

impl Drop for VulkanGemv {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.ctx.vk.queue_wait_idle)(self.ctx.queue);
            (self.ctx.vk.destroy_fence)(self.ctx.device, self.fence, ptr::null());
            (self.ctx.vk.destroy_command_pool)(self.ctx.device, self.command_pool, ptr::null());
            if let Some(pipeline) = self.pipeline.take() {
                pipeline.destroy(&self.ctx);
            }
            if let Some(ref buf) = self.input_buf {
                self.ctx.unmap_buffer(buf);
            }
            if let Some(ref buf) = self.output_buf {
                self.ctx.unmap_buffer(buf);
            }
            if let Some(buf) = self.weight_buf.take() {
                self.ctx.destroy_buffer(buf);
            }
            if let Some(buf) = self.input_buf.take() {
                self.ctx.destroy_buffer(buf);
            }
            if let Some(buf) = self.output_buf.take() {
                self.ctx.destroy_buffer(buf);
            }
        }
    }
}

/// Repack Q8_0 weights from row-major 34B blocks to transposed SoA layout for coalesced GPU access.
///
/// Input: row-major, 34 bytes/block = [f16 scale (2B)] [int8 qs[32] (32B)]
///
/// Output: SoA (Structure of Arrays) transposed layout.
/// For each (block_idx, word_within_block), all rows are contiguous:
///   plane[block * 9 + word][row] = u32 value
///
/// Total words: blocks_per_row * 9 * rows
/// Address: weight[(block * 9 + word) * rows + row]
///
/// This gives perfect coalescing: adjacent threads (adjacent rows) read adjacent u32 values.
pub(crate) fn repack_q8_0_transposed(src: &[u8], rows: usize, blocks_per_row: usize) -> Vec<u32> {
    let src_block_size = 34;
    let words_per_block = 9; // 1 scale word + 8 qs words
    let total_planes = blocks_per_row * words_per_block;
    let mut dst = vec![0u32; total_planes * rows];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let src_off = (row * blocks_per_row + blk) * src_block_size;

            // Scale: 2 bytes → lower 16 bits of u32
            let scale_word = src[src_off] as u32 | ((src[src_off + 1] as u32) << 8);

            // Plane index for scale
            let plane_base = blk * words_per_block;
            dst[plane_base * rows + row] = scale_word;

            // qs: 32 bytes = 8 u32 words
            for w in 0..8 {
                let qs_off = src_off + 2 + w * 4;
                let qs_word = src[qs_off] as u32
                    | ((src[qs_off + 1] as u32) << 8)
                    | ((src[qs_off + 2] as u32) << 16)
                    | ((src[qs_off + 3] as u32) << 24);
                dst[(plane_base + 1 + w) * rows + row] = qs_word;
            }
        }
    }

    dst
}

pub(crate) fn repack_q8_0_transposed_grouped(
    src: &[u8],
    rows: usize,
    blocks_per_row: usize,
    rows_per_group: usize,
) -> Vec<u32> {
    let src_block_size = 34;
    let words_per_block = 9;
    let mut dst = Vec::with_capacity(rows * blocks_per_row * words_per_block);

    let mut group_start = 0usize;
    while group_start < rows {
        let group_rows = (rows - group_start).min(rows_per_group);
        let group_base = dst.len();
        dst.resize(
            group_base + group_rows * blocks_per_row * words_per_block,
            0,
        );

        for local_row in 0..group_rows {
            let row = group_start + local_row;
            for blk in 0..blocks_per_row {
                let src_off = (row * blocks_per_row + blk) * src_block_size;
                let plane_base = blk * words_per_block;
                let scale_word = src[src_off] as u32 | ((src[src_off + 1] as u32) << 8);
                dst[group_base + plane_base * group_rows + local_row] = scale_word;

                for w in 0..8 {
                    let qs_off = src_off + 2 + w * 4;
                    let qs_word = src[qs_off] as u32
                        | ((src[qs_off + 1] as u32) << 8)
                        | ((src[qs_off + 2] as u32) << 16)
                        | ((src[qs_off + 3] as u32) << 24);
                    dst[group_base + (plane_base + 1 + w) * group_rows + local_row] = qs_word;
                }
            }
        }

        group_start += group_rows;
    }

    dst
}

/// Repack Q4_K weights from row-major 144B blocks to transposed SoA layout.
///
/// Q4_K block (144B): [d: 2B (f16)] [dmin: 2B (f16)] [scales: 12B] [qs: 128B]
///
/// SoA planes per block (36 u32 words):
///   plane 0:     d(f16) | dmin(f16) packed as 1 u32
///   plane 1~3:   scales[0..11] as 3 u32
///   plane 4~35:  qs[0..127] as 32 u32
///
/// Address: weight[(block * 36 + plane) * rows + row]
pub(crate) fn repack_q4k_transposed(src: &[u8], rows: usize, blocks_per_row: usize) -> Vec<u32> {
    let src_block_size = 144;
    let words_per_block = 36; // 1 d+dmin + 3 scales + 32 qs
    let total_planes = blocks_per_row * words_per_block;
    let mut dst = vec![0u32; total_planes * rows];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let src_off = (row * blocks_per_row + blk) * src_block_size;
            let plane_base = blk * words_per_block;

            // Plane 0: d(f16, 2B) | dmin(f16, 2B) → 1 u32
            let d_bits = src[src_off] as u32 | ((src[src_off + 1] as u32) << 8);
            let dmin_bits = src[src_off + 2] as u32 | ((src[src_off + 3] as u32) << 8);
            dst[plane_base * rows + row] = d_bits | (dmin_bits << 16);

            // Planes 1~3: scales[0..11] → 3 u32
            for w in 0..3 {
                let s_off = src_off + 4 + w * 4;
                let word = src[s_off] as u32
                    | ((src[s_off + 1] as u32) << 8)
                    | ((src[s_off + 2] as u32) << 16)
                    | ((src[s_off + 3] as u32) << 24);
                dst[(plane_base + 1 + w) * rows + row] = word;
            }

            // Planes 4~35: qs[0..127] → 32 u32
            for w in 0..32 {
                let q_off = src_off + 16 + w * 4;
                let word = src[q_off] as u32
                    | ((src[q_off + 1] as u32) << 8)
                    | ((src[q_off + 2] as u32) << 16)
                    | ((src[q_off + 3] as u32) << 24);
                dst[(plane_base + 4 + w) * rows + row] = word;
            }
        }
    }

    dst
}

/// Repack Q5_K weights from row-major 176B blocks to transposed SoA layout.
///
/// Q5_K block (176B): [d: 2B (f16)] [dmin: 2B (f16)] [scales: 12B] [qh: 32B] [qs: 128B]
///
/// SoA planes per block (44 u32 words):
///   plane 0:       d(f16) | dmin(f16) packed as 1 u32
///   planes 1~3:    scales[0..11] as 3 u32
///   planes 4~35:   qs[0..127] as 32 u32
///   planes 36~43:  qh[0..31] as 8 u32
///
/// Address: weight[(block * 44 + plane) * rows + row]
pub(crate) fn repack_q5k_transposed(src: &[u8], rows: usize, blocks_per_row: usize) -> Vec<u32> {
    let src_block_size = 176;
    let words_per_block = 44;
    let total_planes = blocks_per_row * words_per_block;
    let mut dst = vec![0u32; total_planes * rows];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let src_off = (row * blocks_per_row + blk) * src_block_size;
            let plane_base = blk * words_per_block;

            let d_bits = src[src_off] as u32 | ((src[src_off + 1] as u32) << 8);
            let dmin_bits = src[src_off + 2] as u32 | ((src[src_off + 3] as u32) << 8);
            dst[plane_base * rows + row] = d_bits | (dmin_bits << 16);

            for w in 0..3 {
                let s_off = src_off + 4 + w * 4;
                let word = src[s_off] as u32
                    | ((src[s_off + 1] as u32) << 8)
                    | ((src[s_off + 2] as u32) << 16)
                    | ((src[s_off + 3] as u32) << 24);
                dst[(plane_base + 1 + w) * rows + row] = word;
            }

            for w in 0..32 {
                let q_off = src_off + 48 + w * 4;
                let word = src[q_off] as u32
                    | ((src[q_off + 1] as u32) << 8)
                    | ((src[q_off + 2] as u32) << 16)
                    | ((src[q_off + 3] as u32) << 24);
                dst[(plane_base + 4 + w) * rows + row] = word;
            }

            for w in 0..8 {
                let h_off = src_off + 16 + w * 4;
                let word = src[h_off] as u32
                    | ((src[h_off + 1] as u32) << 8)
                    | ((src[h_off + 2] as u32) << 16)
                    | ((src[h_off + 3] as u32) << 24);
                dst[(plane_base + 36 + w) * rows + row] = word;
            }
        }
    }

    dst
}

/// Repack Q6_K weights from row-major 210B blocks to transposed SoA layout.
///
/// Q6_K block (210B): [ql: 128B] [qh: 64B] [scales: 16B (i8)] [d: 2B (f16)]
///
/// SoA planes per block (53 u32 words):
///   plane 0:       d(f16) in lower 16 bits, upper 16 bits = 0
///   planes 1~4:    scales[0..15] as 4 u32 (signed i8 values packed as bytes)
///   planes 5~36:   ql[0..127] as 32 u32
///   planes 37~52:  qh[0..63] as 16 u32
///
/// Address: weight[(block * 53 + plane) * rows + row]
pub(crate) fn repack_q6k_transposed(src: &[u8], rows: usize, blocks_per_row: usize) -> Vec<u32> {
    let src_block_size = 210;
    let words_per_block = 53; // 1 d + 4 scales + 32 ql + 16 qh
    let total_planes = blocks_per_row * words_per_block;
    let mut dst = vec![0u32; total_planes * rows];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let src_off = (row * blocks_per_row + blk) * src_block_size;
            let plane_base = blk * words_per_block;

            // Plane 0: d(f16, 2B) at offset 208-209
            let d_off = src_off + 208;
            let d_bits = src[d_off] as u32 | ((src[d_off + 1] as u32) << 8);
            dst[plane_base * rows + row] = d_bits;

            // Planes 1~4: scales[0..15] at offset 192-207 → 4 u32
            for w in 0..4 {
                let s_off = src_off + 192 + w * 4;
                let word = src[s_off] as u32
                    | ((src[s_off + 1] as u32) << 8)
                    | ((src[s_off + 2] as u32) << 16)
                    | ((src[s_off + 3] as u32) << 24);
                dst[(plane_base + 1 + w) * rows + row] = word;
            }

            // Planes 5~36: ql[0..127] at offset 0-127 → 32 u32
            for w in 0..32 {
                let q_off = src_off + w * 4;
                let word = src[q_off] as u32
                    | ((src[q_off + 1] as u32) << 8)
                    | ((src[q_off + 2] as u32) << 16)
                    | ((src[q_off + 3] as u32) << 24);
                dst[(plane_base + 5 + w) * rows + row] = word;
            }

            // Planes 37~52: qh[0..63] at offset 128-191 → 16 u32
            for w in 0..16 {
                let h_off = src_off + 128 + w * 4;
                let word = src[h_off] as u32
                    | ((src[h_off + 1] as u32) << 8)
                    | ((src[h_off + 2] as u32) << 16)
                    | ((src[h_off + 3] as u32) << 24);
                dst[(plane_base + 37 + w) * rows + row] = word;
            }
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU-side Q4_K dequantization (standalone, no rnb-cpu dependency)
    fn cpu_extract_scales(sb: &[u8; 12]) -> ([f32; 8], [f32; 8]) {
        let mut sc = [0f32; 8];
        let mut mn = [0f32; 8];
        for j in 0..8usize {
            let (s, m) = if j < 4 {
                (sb[j] & 63, sb[j + 4] & 63)
            } else {
                let s = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                let m = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                (s, m)
            };
            sc[j] = s as f32;
            mn[j] = m as f32;
        }
        (sc, mn)
    }

    fn cpu_dequant_q4k(block: &[u8; 144]) -> [f32; 256] {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; 12] = block[4..16].try_into().unwrap();
        let qs = &block[16..144];

        let (sc, mn) = cpu_extract_scales(&scales);
        let mut out = [0f32; 256];

        let mut is = 0;
        let mut q_off = 0;
        let mut y_off = 0;
        for _ in 0..4 {
            let d1 = d * sc[is];
            let m1 = dmin * mn[is];
            let d2 = d * sc[is + 1];
            let m2 = dmin * mn[is + 1];
            for l in 0..32 {
                out[y_off + l] = d1 * (qs[q_off + l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                out[y_off + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
            q_off += 32;
            is += 2;
            y_off += 64;
        }
        out
    }

    fn cpu_dequant_q5k(block: &[u8; 176]) -> [f32; 256] {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; 12] = block[4..16].try_into().unwrap();
        let qh = &block[16..48];
        let qs = &block[48..176];

        let (sc, mn) = cpu_extract_scales(&scales);
        let mut out = [0f32; 256];

        let mut is = 0usize;
        let mut ql_off = 0usize;
        let mut y_off = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _ in 0..4 {
            let d1 = d * sc[is];
            let m1 = dmin * mn[is];
            let d2 = d * sc[is + 1];
            let m2 = dmin * mn[is + 1];

            for l in 0..32 {
                let high = if qh[l] & u1 != 0 { 16 } else { 0 };
                out[y_off + l] = d1 * ((qs[ql_off + l] & 0xF) + high) as f32 - m1;
            }
            for l in 0..32 {
                let high = if qh[l] & u2 != 0 { 16 } else { 0 };
                out[y_off + 32 + l] = d2 * ((qs[ql_off + l] >> 4) + high) as f32 - m2;
            }

            ql_off += 32;
            is += 2;
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
            y_off += 64;
        }
        out
    }

    /// GPU shader f16→f32 conversion (matching SPIR-V with proper denormal handling)
    fn gpu_f16_to_f32(raw: u32) -> f32 {
        let exp = (raw >> 10) & 0x1F;
        let mant = raw & 0x3FF;
        if exp == 0 {
            // Denormal or zero: value = mant * 2^(-24)
            return mant as f32 * 5.9604644775390625e-8;
        }
        // Normal
        let sign_bit = (raw >> 15) & 1;
        let s_part = sign_bit << 31;
        let e_adj = exp + 112;
        let e_part = e_adj << 23;
        let m_part = mant << 13;
        let bits = s_part | e_part | m_part;
        f32::from_bits(bits)
    }

    /// Simulate the GPU shader's dequant+dot logic using repacked data
    fn gpu_simulate_q4k_dot(
        repacked: &[u32],
        row: usize,
        rows: usize,
        input: &[f32],
        blocks_per_row: usize,
    ) -> f32 {
        let mut total_sum = 0.0f32;

        for blk in 0..blocks_per_row {
            let plane_base = blk * 36 * rows;

            // Load d, dmin from plane 0
            let packed_word = repacked[plane_base + row];
            let d_raw = packed_word & 0xFFFF;
            let dmin_raw = packed_word >> 16;
            let d = gpu_f16_to_f32(d_raw);
            let dmin = gpu_f16_to_f32(dmin_raw);

            // Load scale bytes from planes 1-3
            let s0_word = repacked[(blk * 36 + 1) * rows + row];
            let s1_word = repacked[(blk * 36 + 2) * rows + row];
            let s2_word = repacked[(blk * 36 + 3) * rows + row];

            let mut sb = [0u32; 12];
            for i in 0..4u32 {
                sb[i as usize] = (s0_word >> (i * 8)) & 0xFF;
            }
            for i in 0..4u32 {
                sb[4 + i as usize] = (s1_word >> (i * 8)) & 0xFF;
            }
            for i in 0..4u32 {
                sb[8 + i as usize] = (s2_word >> (i * 8)) & 0xFF;
            }

            // Extract scales and mins (matching shader logic)
            let mut scales = [0u32; 8];
            let mut mins = [0u32; 8];
            for j in 0..4usize {
                scales[j] = sb[j] & 0x3F;
                mins[j] = sb[j + 4] & 0x3F;
            }
            for j in 4..8usize {
                scales[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                mins[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
            }

            // Process 8 sub-blocks
            let blk_x_256 = blk * 256;
            for sb_idx in 0..8u32 {
                let sc_f = scales[sb_idx as usize] as f32;
                let mn_f = mins[sb_idx as usize] as f32;

                let qs_group = sb_idx / 2;
                let is_high = (sb_idx & 1) != 0;
                let qs_base_plane = 4 + qs_group * 8;

                let inp_sb_base = blk_x_256 + (sb_idx as usize) * 32;

                let mut nibble_input_sum = 0.0f32;
                let mut input_sum = 0.0f32;

                for w in 0..8u32 {
                    let qs_plane = blk * 36 + qs_base_plane as usize + w as usize;
                    let qs_word = repacked[qs_plane * rows + row];

                    for byte_idx in 0..4u32 {
                        let shift_amt = if is_high {
                            byte_idx * 8 + 4
                        } else {
                            byte_idx * 8
                        };
                        let nibble = (qs_word >> shift_amt) & 0x0F;
                        let nibble_f = nibble as f32;

                        let inp_idx = inp_sb_base + (w * 4 + byte_idx) as usize;
                        let inp_val = input[inp_idx];

                        nibble_input_sum += nibble_f * inp_val;
                        input_sum += inp_val;
                    }
                }

                let d_sc = d * sc_f;
                let term1 = d_sc * nibble_input_sum;
                let dmin_mn = dmin * mn_f;
                let term2 = dmin_mn * input_sum;
                total_sum += term1 - term2;
            }
        }

        total_sum
    }

    fn gpu_simulate_q5k_dot(
        repacked: &[u32],
        row: usize,
        rows: usize,
        input: &[f32],
        blocks_per_row: usize,
    ) -> f32 {
        let mut total_sum = 0.0f32;

        for blk in 0..blocks_per_row {
            let plane_base = blk * 44 * rows;

            let packed_word = repacked[plane_base + row];
            let d_raw = packed_word & 0xFFFF;
            let dmin_raw = packed_word >> 16;
            let d = gpu_f16_to_f32(d_raw);
            let dmin = gpu_f16_to_f32(dmin_raw);

            let s0_word = repacked[(blk * 44 + 1) * rows + row];
            let s1_word = repacked[(blk * 44 + 2) * rows + row];
            let s2_word = repacked[(blk * 44 + 3) * rows + row];

            let mut sb = [0u32; 12];
            for i in 0..4u32 {
                sb[i as usize] = (s0_word >> (i * 8)) & 0xFF;
                sb[4 + i as usize] = (s1_word >> (i * 8)) & 0xFF;
                sb[8 + i as usize] = (s2_word >> (i * 8)) & 0xFF;
            }

            let mut scales = [0u32; 8];
            let mut mins = [0u32; 8];
            for j in 0..4usize {
                scales[j] = sb[j] & 0x3F;
                mins[j] = sb[j + 4] & 0x3F;
            }
            for j in 4..8usize {
                scales[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                mins[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
            }

            let blk_x_256 = blk * 256;
            for sb_idx in 0..8u32 {
                let sc_f = scales[sb_idx as usize] as f32;
                let mn_f = mins[sb_idx as usize] as f32;
                let qs_group = sb_idx / 2;
                let is_high = (sb_idx & 1) != 0;
                let qs_base_plane = 4 + qs_group * 8;
                let inp_sb_base = blk_x_256 + (sb_idx as usize) * 32;

                let mut q_input_sum = 0.0f32;
                let mut input_sum = 0.0f32;

                for w in 0..8u32 {
                    let qs_plane = blk * 44 + qs_base_plane as usize + w as usize;
                    let qs_word = repacked[qs_plane * rows + row];
                    let qh_plane = blk * 44 + 36 + w as usize;
                    let qh_word = repacked[qh_plane * rows + row];

                    for byte_idx in 0..4u32 {
                        let shift_amt = if is_high {
                            byte_idx * 8 + 4
                        } else {
                            byte_idx * 8
                        };
                        let nibble = (qs_word >> shift_amt) & 0x0F;
                        let qh_shift = byte_idx * 8 + sb_idx;
                        let high = ((qh_word >> qh_shift) & 0x01) << 4;
                        let q5 = (nibble | high) as f32;

                        let inp_idx = inp_sb_base + (w * 4 + byte_idx) as usize;
                        let inp_val = input[inp_idx];
                        q_input_sum += q5 * inp_val;
                        input_sum += inp_val;
                    }
                }

                let term1 = d * sc_f * q_input_sum;
                let term2 = dmin * mn_f * input_sum;
                total_sum += term1 - term2;
            }
        }

        total_sum
    }

    #[test]
    fn q8_0_grouped_repack_uses_group_local_row_stride() {
        let rows = 5usize;
        let blocks_per_row = 2usize;
        let rows_per_group = 3usize;
        let mut raw = Vec::with_capacity(rows * blocks_per_row * 34);
        for row in 0..rows {
            for blk in 0..blocks_per_row {
                raw.push((row * 10 + blk) as u8);
                raw.push((row * 10 + blk + 1) as u8);
                for elem in 0..32usize {
                    raw.push((row * 40 + blk * 7 + elem) as u8);
                }
            }
        }

        let grouped = repack_q8_0_transposed_grouped(&raw, rows, blocks_per_row, rows_per_group);

        let words_per_block = 9usize;
        let first_group_words = rows_per_group * blocks_per_row * words_per_block;
        let second_group_rows = 2usize;
        assert_eq!(
            grouped.len(),
            first_group_words + second_group_rows * blocks_per_row * words_per_block
        );

        let row = 4usize;
        let local_row = 1usize;
        let group_base = first_group_words;
        let src_off = (row * blocks_per_row + 1) * 34;
        let scale = raw[src_off] as u32 | ((raw[src_off + 1] as u32) << 8);
        assert_eq!(
            grouped[group_base + (words_per_block * second_group_rows) + local_row],
            scale
        );

        let qs_word = raw[src_off + 2] as u32
            | ((raw[src_off + 3] as u32) << 8)
            | ((raw[src_off + 4] as u32) << 16)
            | ((raw[src_off + 5] as u32) << 24);
        assert_eq!(
            grouped[group_base + ((words_per_block + 1) * second_group_rows) + local_row],
            qs_word
        );
    }

    #[test]
    fn test_q4k_shader_simulation_matches_cpu() {
        let mut block = [0u8; 144];

        // d = 0.05 as f16, dmin = 0.02 as f16
        let d_f16 = half::f16::from_f32(0.05);
        let dmin_f16 = half::f16::from_f32(0.02);
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        block[2..4].copy_from_slice(&dmin_f16.to_le_bytes());

        // scales: diverse 6-bit values
        let scale_bytes: [u8; 12] = [
            0b00_110100, // byte 0: sc[0]=52
            0b01_010110, // byte 1: sc[1]=22
            0b10_101010, // byte 2: sc[2]=42
            0b11_001111, // byte 3: sc[3]=15
            0b01_011001, // byte 4: mn[0]=25
            0b10_100011, // byte 5: mn[1]=35
            0b00_010101, // byte 6: mn[2]=21
            0b11_111000, // byte 7: mn[3]=56
            0b1010_0111, // byte 8: sc[4]_lo=7, mn[4]_lo=10
            0b0011_1100, // byte 9: sc[5]_lo=12, mn[5]_lo=3
            0b1110_0010, // byte 10: sc[6]_lo=2, mn[6]_lo=14
            0b0101_1001, // byte 11: sc[7]_lo=9, mn[7]_lo=5
        ];
        block[4..16].copy_from_slice(&scale_bytes);

        // qs: varied nibble values
        for i in 0..128 {
            let lo = ((i * 7 + 3) % 16) as u8;
            let hi = ((i * 11 + 5) % 16) as u8;
            block[16 + i] = lo | (hi << 4);
        }

        // Input vector (256 elements)
        let mut input = vec![0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.37).sin() * 2.0) + 0.5;
        }

        // CPU: dequant then dot product
        let dequantized = cpu_dequant_q4k(&block);
        let cpu_dot: f32 = dequantized
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();

        // GPU simulation: repack then simulate shader
        let repacked = repack_q4k_transposed(&block, 1, 1);
        let gpu_dot = gpu_simulate_q4k_dot(&repacked, 0, 1, &input, 1);

        let diff = (cpu_dot - gpu_dot).abs();
        eprintln!("CPU dot:  {cpu_dot}");
        eprintln!("GPU dot:  {gpu_dot}");
        eprintln!("Diff:     {diff}");

        // f16 conversion comparison
        let cpu_d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let gpu_d = gpu_f16_to_f32(block[0] as u32 | ((block[1] as u32) << 8));
        eprintln!("d CPU={cpu_d}, GPU={gpu_d}, diff={}", (cpu_d - gpu_d).abs());

        let cpu_dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let gpu_dmin = gpu_f16_to_f32(block[2] as u32 | ((block[3] as u32) << 8));
        eprintln!(
            "dmin CPU={cpu_dmin}, GPU={gpu_dmin}, diff={}",
            (cpu_dmin - gpu_dmin).abs()
        );

        // Scale comparison
        let (cpu_sc, cpu_mn) = cpu_extract_scales(&scale_bytes);
        eprintln!("CPU scales: {:?}", cpu_sc);
        eprintln!("CPU mins:   {:?}", cpu_mn);

        assert!(
            diff < 0.01,
            "Q4_K shader simulation mismatch: CPU={cpu_dot}, GPU={gpu_dot}, diff={diff}"
        );
    }

    #[test]
    fn test_f16_conversion() {
        // Zero
        let gpu_zero = gpu_f16_to_f32(0);
        assert_eq!(gpu_zero, 0.0, "f16 zero should be 0.0");

        // Normal value
        let test_val = half::f16::from_f32(0.05);
        let raw = test_val.to_bits() as u32;
        let gpu = gpu_f16_to_f32(raw);
        let cpu = test_val.to_f32();
        assert!(
            (cpu - gpu).abs() < 1e-10,
            "normal f16: CPU={cpu}, GPU={gpu}"
        );

        // Denormal: 0x03eb (from real model, d=5.978e-5)
        let denorm_raw = 0x03ebu32;
        let cpu_denorm = half::f16::from_bits(denorm_raw as u16).to_f32();
        let gpu_denorm = gpu_f16_to_f32(denorm_raw);
        let denorm_err = ((gpu_denorm - cpu_denorm) / cpu_denorm).abs();
        eprintln!(
            "Denormal 0x03eb: cpu={:.8}, gpu={:.8}, rel_err={:.4}%",
            cpu_denorm,
            gpu_denorm,
            denorm_err * 100.0
        );
        // ~1% error acceptable for denormals
        assert!(
            denorm_err < 0.02,
            "denormal error too large: {:.4}%",
            denorm_err * 100.0
        );
        assert!(
            gpu_denorm > 0.0,
            "denormal should NOT be zero: {}",
            gpu_denorm
        );
    }

    #[test]
    fn test_q4k_with_real_model_scales() {
        // Exact bytes from Qwen3.5-0.8B L3 q_proj row 0
        let mut block = [0u8; 144];
        // d=0x04ba (0.000072), dmin=0x1102 (0.000611)
        block[0] = 0xba;
        block[1] = 0x04;
        block[2] = 0x02;
        block[3] = 0x11;
        // scales: bf aa b3 f1 bf ab aa a6 3f 36 9c 80
        let scales_raw = [
            0xbf, 0xaa, 0xb3, 0xf1, 0xbf, 0xab, 0xaa, 0xa6, 0x3f, 0x36, 0x9c, 0x80u8,
        ];
        block[4..16].copy_from_slice(&scales_raw);

        // qs: use diverse pattern (we don't have real qs, use deterministic)
        for i in 0..128 {
            let lo = ((i * 7 + 3) % 16) as u8;
            let hi = ((i * 11 + 5) % 16) as u8;
            block[16 + i] = lo | (hi << 4);
        }

        // Input: varied values
        let mut input = vec![0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.37).sin() * 2.0) + 0.5;
        }

        let dequantized = cpu_dequant_q4k(&block);
        let cpu_dot: f32 = dequantized
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();

        let repacked = repack_q4k_transposed(&block, 1, 1);
        let gpu_dot = gpu_simulate_q4k_dot(&repacked, 0, 1, &input, 1);

        let diff = (cpu_dot - gpu_dot).abs();
        eprintln!("Real scales: CPU dot={cpu_dot}, GPU dot={gpu_dot}, diff={diff}");

        // Also print extracted scales for verification
        let (sc, mn) = cpu_extract_scales(&scales_raw);
        eprintln!("Extracted scales: {:?}", sc);
        eprintln!("Extracted mins:   {:?}", mn);

        assert!(
            diff < 0.01,
            "Real model scales mismatch: CPU={cpu_dot}, GPU={gpu_dot}, diff={diff}"
        );
    }

    /// CPU Q6_K dequant (standalone)
    fn cpu_dequant_q6k(block: &[u8; 210]) -> [f32; 256] {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208]; // i8
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
        let mut out = [0f32; 256];

        for n in 0..2 {
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;
            let y_base = n * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = (ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4);
                let q2 = (ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4);
                let q3 = (ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4);
                let q4 = (ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4);
                out[y_base + l] = d * (sc[sc_base + is] as i8 as f32) * (q1 as i32 - 32) as f32;
                out[y_base + l + 32] =
                    d * (sc[sc_base + is + 2] as i8 as f32) * (q2 as i32 - 32) as f32;
                out[y_base + l + 64] =
                    d * (sc[sc_base + is + 4] as i8 as f32) * (q3 as i32 - 32) as f32;
                out[y_base + l + 96] =
                    d * (sc[sc_base + is + 6] as i8 as f32) * (q4 as i32 - 32) as f32;
            }
        }
        out
    }

    /// Simulate Q6_K GPU shader logic using repacked data
    fn gpu_simulate_q6k_dot(
        repacked: &[u32],
        row: usize,
        rows: usize,
        input: &[f32],
        blocks_per_row: usize,
    ) -> f32 {
        let mut total_sum = 0.0f32;
        for blk in 0..blocks_per_row {
            let plane_base = blk * 53 * rows;

            // Load d from plane 0
            let packed = repacked[plane_base + row];
            let d_raw = packed & 0xFFFF;
            let d = gpu_f16_to_f32(d_raw);

            // Load scale words from planes 1-4
            let mut sw = [0u32; 4];
            for i in 0..4 {
                sw[i] = repacked[(blk * 53 + 1 + i) * rows + row];
            }

            struct SB {
                ql_off: usize,
                hi: bool,
                qh_off: usize,
                qh_shift: u32,
                elem_off: usize,
                sc_idx: usize,
            }
            let sbs = [
                SB {
                    ql_off: 5,
                    hi: false,
                    qh_off: 37,
                    qh_shift: 0,
                    elem_off: 0,
                    sc_idx: 0,
                },
                SB {
                    ql_off: 9,
                    hi: false,
                    qh_off: 41,
                    qh_shift: 0,
                    elem_off: 16,
                    sc_idx: 1,
                },
                SB {
                    ql_off: 13,
                    hi: false,
                    qh_off: 37,
                    qh_shift: 2,
                    elem_off: 32,
                    sc_idx: 2,
                },
                SB {
                    ql_off: 17,
                    hi: false,
                    qh_off: 41,
                    qh_shift: 2,
                    elem_off: 48,
                    sc_idx: 3,
                },
                SB {
                    ql_off: 5,
                    hi: true,
                    qh_off: 37,
                    qh_shift: 4,
                    elem_off: 64,
                    sc_idx: 4,
                },
                SB {
                    ql_off: 9,
                    hi: true,
                    qh_off: 41,
                    qh_shift: 4,
                    elem_off: 80,
                    sc_idx: 5,
                },
                SB {
                    ql_off: 13,
                    hi: true,
                    qh_off: 37,
                    qh_shift: 6,
                    elem_off: 96,
                    sc_idx: 6,
                },
                SB {
                    ql_off: 17,
                    hi: true,
                    qh_off: 41,
                    qh_shift: 6,
                    elem_off: 112,
                    sc_idx: 7,
                },
                SB {
                    ql_off: 21,
                    hi: false,
                    qh_off: 45,
                    qh_shift: 0,
                    elem_off: 128,
                    sc_idx: 8,
                },
                SB {
                    ql_off: 25,
                    hi: false,
                    qh_off: 49,
                    qh_shift: 0,
                    elem_off: 144,
                    sc_idx: 9,
                },
                SB {
                    ql_off: 29,
                    hi: false,
                    qh_off: 45,
                    qh_shift: 2,
                    elem_off: 160,
                    sc_idx: 10,
                },
                SB {
                    ql_off: 33,
                    hi: false,
                    qh_off: 49,
                    qh_shift: 2,
                    elem_off: 176,
                    sc_idx: 11,
                },
                SB {
                    ql_off: 21,
                    hi: true,
                    qh_off: 45,
                    qh_shift: 4,
                    elem_off: 192,
                    sc_idx: 12,
                },
                SB {
                    ql_off: 25,
                    hi: true,
                    qh_off: 49,
                    qh_shift: 4,
                    elem_off: 208,
                    sc_idx: 13,
                },
                SB {
                    ql_off: 29,
                    hi: true,
                    qh_off: 45,
                    qh_shift: 6,
                    elem_off: 224,
                    sc_idx: 14,
                },
                SB {
                    ql_off: 33,
                    hi: true,
                    qh_off: 49,
                    qh_shift: 6,
                    elem_off: 240,
                    sc_idx: 15,
                },
            ];

            let blk_x_256 = blk * 256;
            for sb in &sbs {
                // Extract signed i8 scale
                let sw_idx = sb.sc_idx / 4;
                let byte_shift = (sb.sc_idx % 4) * 8;
                let byte_val = ((sw[sw_idx] >> byte_shift) & 0xFF) as u8;
                let scale_f = byte_val as i8 as f32;
                let d_scale = d * scale_f;

                for w in 0..4u32 {
                    let ql_plane = blk * 53 + sb.ql_off + w as usize;
                    let ql_word = repacked[ql_plane * rows + row];
                    let qh_plane = blk * 53 + sb.qh_off + w as usize;
                    let qh_word = repacked[qh_plane * rows + row];

                    for bi in 0..4u32 {
                        let ql_shift = if sb.hi { bi * 8 + 4 } else { bi * 8 };
                        let ql_nibble = (ql_word >> ql_shift) & 0x0F;

                        let qh_shift = bi * 8 + sb.qh_shift;
                        let qh_bits = (qh_word >> qh_shift) & 0x3;

                        let q6 = ql_nibble | (qh_bits << 4);
                        let q6_centered = q6 as i32 - 32;

                        let inp_idx = blk_x_256 + sb.elem_off + (w * 4 + bi) as usize;
                        total_sum += d_scale * q6_centered as f32 * input[inp_idx];
                    }
                }
            }
        }
        total_sum
    }

    #[test]
    fn test_q6k_shader_simulation_matches_cpu() {
        let mut block = [0u8; 210];
        // ql: varied nibble values
        for i in 0..128 {
            block[i] = ((i * 7 + 3) % 256) as u8;
        }
        // qh: varied 2-bit values
        for i in 0..64 {
            block[128 + i] = ((i * 13 + 5) % 256) as u8;
        }
        // scales: signed i8 values
        let scales: [i8; 16] = [
            10, -5, 20, -15, 8, -3, 25, -10, 12, -7, 18, -20, 6, -1, 30, -8,
        ];
        for i in 0..16 {
            block[192 + i] = scales[i] as u8;
        }
        // d = 0.05 as f16
        let d_f16 = half::f16::from_f32(0.05);
        block[208..210].copy_from_slice(&d_f16.to_le_bytes());

        let mut input = vec![0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.37).sin() * 2.0) + 0.5;
        }

        let dequantized = cpu_dequant_q6k(&block);
        let cpu_dot: f32 = dequantized
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();

        let repacked = repack_q6k_transposed(&block, 1, 1);
        let gpu_dot = gpu_simulate_q6k_dot(&repacked, 0, 1, &input, 1);

        let diff = (cpu_dot - gpu_dot).abs();
        eprintln!("Q6_K: CPU dot={cpu_dot}, GPU dot={gpu_dot}, diff={diff}");

        assert!(diff < 0.01, "Q6_K shader simulation mismatch: diff={diff}");
    }

    #[test]
    fn test_q5k_shader_simulation_matches_cpu() {
        let mut block = [0u8; 176];

        let d_f16 = half::f16::from_f32(0.05);
        let dmin_f16 = half::f16::from_f32(0.02);
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        block[2..4].copy_from_slice(&dmin_f16.to_le_bytes());

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
            block[16 + i] = ((i * 13 + 7) % 256) as u8;
        }
        for i in 0..128 {
            let lo = ((i * 7 + 3) % 16) as u8;
            let hi = ((i * 11 + 5) % 16) as u8;
            block[48 + i] = lo | (hi << 4);
        }

        let mut input = vec![0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.37).sin() * 2.0) + 0.5;
        }

        let dequantized = cpu_dequant_q5k(&block);
        let cpu_dot: f32 = dequantized
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();

        let repacked = repack_q5k_transposed(&block, 1, 1);
        let gpu_dot = gpu_simulate_q5k_dot(&repacked, 0, 1, &input, 1);
        let diff = (cpu_dot - gpu_dot).abs();

        eprintln!("Q5_K: CPU dot={cpu_dot}, GPU dot={gpu_dot}, diff={diff}");
        assert!(diff < 0.01, "Q5_K shader simulation mismatch: diff={diff}");
    }

    #[test]
    fn test_q5k_repack_layout_basic() {
        let mut block = [0u8; 176];
        block[0] = 0x34;
        block[1] = 0x12;
        block[2] = 0x78;
        block[3] = 0x56;
        for i in 0..12 {
            block[4 + i] = i as u8;
        }
        for i in 0..32 {
            block[16 + i] = (0x80 + i) as u8;
        }
        for i in 0..128 {
            block[48 + i] = (0x40 + i) as u8;
        }

        let repacked = repack_q5k_transposed(&block, 1, 1);
        assert_eq!(repacked.len(), 44);
        assert_eq!(repacked[0], 0x5678_1234);
        assert_eq!(repacked[1], 0x0302_0100);
        assert_eq!(repacked[3], 0x0b0a_0908);
        assert_eq!(repacked[4], 0x4342_4140);
        assert_eq!(repacked[35], 0xbfbe_bdbc);
        assert_eq!(repacked[36], 0x8382_8180);
        assert_eq!(repacked[43], 0x9f9e_9d9c);
    }

    #[test]
    fn test_q4k_zero_d_block() {
        // Block with d=0.0 (all quant values should be zero)
        let mut block = [0u8; 144];
        // d = 0.0, dmin = 0.02
        block[0..2].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
        // Non-zero scales and qs
        for i in 0..12 {
            block[4 + i] = 0x15;
        }
        for i in 0..128 {
            block[16 + i] = 0xAB;
        }

        let mut input = vec![1.0f32; 256];
        for i in 0..256 {
            input[i] = (i as f32 * 0.1).sin() + 0.5;
        }

        let dequantized = cpu_dequant_q4k(&block);
        let cpu_dot: f32 = dequantized
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();

        let repacked = repack_q4k_transposed(&block, 1, 1);
        let gpu_dot = gpu_simulate_q4k_dot(&repacked, 0, 1, &input, 1);

        let diff = (cpu_dot - gpu_dot).abs();
        eprintln!("d=0 block: CPU={cpu_dot}, GPU={gpu_dot}, diff={diff}");
        assert!(
            diff < 0.01,
            "d=0 block mismatch: CPU={cpu_dot}, GPU={gpu_dot}, diff={diff}"
        );
    }

    // ---------------------------------------------------------------------
    // Q*K × Q8K shader simulation tests (mv31 0순위 — CPU strict parity path)
    //
    // The Q4_K × Q8K and Q6_K × Q8K integer-dot shaders mirror CPU strict
    // `dot_q4_k_q8k_scalar` / `dot_q6_k_q8k_scalar` semantics. These tests
    // build a single block of weights, run the same input through both the
    // CPU integer-dot reference and a pure-Rust mirror of the shader, and
    // require bit-for-bit equivalence (exact `f32` equality, not epsilon).
    // ---------------------------------------------------------------------

    /// Quantize a single 256-element f32 chunk into the Q8K integer block layout.
    /// Mirrors `quantize_input_q8k` from rnb-cpu (no aarch64 dependency).
    fn quantize_q8k_block(input: &[f32; 256]) -> ([i8; 256], f32, [i16; 8]) {
        let mut amax = 0.0f32;
        for &x in input.iter() {
            amax = amax.max(x.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 8];
        for i in 0..256 {
            let q = (input[i] * id).round().clamp(-128.0, 127.0) as i8;
            qs[i] = q;
            bsums[i / 32] += q as i16;
        }
        (qs, d, bsums)
    }

    /// Pack a single Q8K block into the shader binding format
    /// (69 u32 words: 64 qs words + 1 d word + 4 bsums words).
    fn pack_q8k_for_shader(qs: &[i8; 256], d: f32, bsums: &[i16; 8]) -> Vec<u32> {
        let mut packed = vec![0u32; 69];
        // words 0..64: qs (4 i8 per u32, little-endian)
        for w in 0..64 {
            let mut word = 0u32;
            for byte in 0..4 {
                let b = qs[w * 4 + byte] as u8 as u32;
                word |= b << (byte * 8);
            }
            packed[w] = word;
        }
        // word 64: d as f32 bits
        packed[64] = d.to_bits();
        // words 65..69: bsums (2 i16 per u32, low half then high half)
        for g in 0..4 {
            let lo = bsums[g * 2] as u16 as u32;
            let hi = bsums[g * 2 + 1] as u16 as u32;
            packed[65 + g] = lo | (hi << 16);
        }
        packed
    }

    /// CPU reference for a single Q4_K block × Q8K activation block.
    /// Mirrors `dot_q4_k_q8k_scalar` from rnb-cpu/src/gemm/neon_dot.rs.
    fn cpu_q4k_q8k_integer_dot(
        block: &[u8; 144],
        qs: &[i8; 256],
        q8k_d: f32,
        bsums: &[i16; 8],
    ) -> f32 {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales: [u8; 12] = block[4..16].try_into().unwrap();
        let qs_w = &block[16..144];

        // Extract scales/mins as raw u8 (matching dot_q4_k_q8k_scalar's extract_q4_k_scales_mins).
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..8usize {
            if j < 4 {
                sc[j] = scales[j] & 0x3F;
                mn[j] = scales[j + 4] & 0x3F;
            } else {
                sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }
        }

        let mut sumi = 0i32;
        let mut summ = 0i32;
        for group in 0..4 {
            let q_off = group * 32;
            let x_off = group * 64;
            let is = group * 2;
            let mut isum0 = 0i32;
            let mut isum1 = 0i32;
            for l in 0..32 {
                let lo = (qs_w[q_off + l] & 0x0F) as i8;
                let hi = (qs_w[q_off + l] >> 4) as i8;
                isum0 += lo as i32 * qs[x_off + l] as i32;
                isum1 += hi as i32 * qs[x_off + 32 + l] as i32;
            }
            sumi += sc[is] as i32 * isum0 + sc[is + 1] as i32 * isum1;
            summ += mn[is] as i32 * bsums[group * 2] as i32
                + mn[is + 1] as i32 * bsums[group * 2 + 1] as i32;
        }
        q8k_d * (d * sumi as f32 - dmin * summ as f32)
    }

    /// CPU reference for a single Q6_K block × Q8K activation block.
    /// Mirrors `dot_q6_k_q8k_scalar` from rnb-cpu/src/gemm/neon_dot.rs.
    fn cpu_q6k_q8k_integer_dot(block: &[u8; 210], qs: &[i8; 256], q8k_d: f32) -> f32 {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

        let mut sumi = 0i32;
        for n in 0..2 {
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;
            let x_base = n * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 =
                    ((ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4))
                    as i32
                    - 32;
                let q3 = ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i32 - 32;
                let sc1 = scales[sc_base + is] as i8 as i32;
                let sc2 = scales[sc_base + is + 2] as i8 as i32;
                let sc3 = scales[sc_base + is + 4] as i8 as i32;
                let sc4 = scales[sc_base + is + 6] as i8 as i32;
                sumi += sc1 * q1 * qs[x_base + l] as i32;
                sumi += sc2 * q2 * qs[x_base + l + 32] as i32;
                sumi += sc3 * q3 * qs[x_base + l + 64] as i32;
                sumi += sc4 * q4 * qs[x_base + l + 96] as i32;
            }
        }
        d * q8k_d * sumi as f32
    }

    /// Pure-Rust mirror of `emit_q4k_q8k_gemv` shader logic.
    /// Operates on the same transposed weight repack as `gpu_simulate_q4k_dot`,
    /// but consumes Q8K activation packed in the shader binding format (69 u32/block).
    fn gpu_simulate_q4k_q8k_dot(
        repacked: &[u32],
        row: usize,
        rows: usize,
        packed_act: &[u32],
        blocks_per_row: usize,
    ) -> f32 {
        let mut total = 0.0f32;
        for blk in 0..blocks_per_row {
            let plane_base = blk * 36 * rows;
            let act_base = blk * 69;

            // d, dmin from plane 0 (low 16 = d_f16, high 16 = dmin_f16)
            let packed_word = repacked[plane_base + row];
            let d_raw = packed_word & 0xFFFF;
            let dmin_raw = packed_word >> 16;
            let d = gpu_f16_to_f32(d_raw);
            let dmin = gpu_f16_to_f32(dmin_raw);

            // 12 scales/mins bytes from planes 1..3
            let s0_word = repacked[(blk * 36 + 1) * rows + row];
            let s1_word = repacked[(blk * 36 + 2) * rows + row];
            let s2_word = repacked[(blk * 36 + 3) * rows + row];
            let mut sb = [0u32; 12];
            for i in 0..4u32 {
                sb[i as usize] = (s0_word >> (i * 8)) & 0xFF;
            }
            for i in 0..4u32 {
                sb[4 + i as usize] = (s1_word >> (i * 8)) & 0xFF;
            }
            for i in 0..4u32 {
                sb[8 + i as usize] = (s2_word >> (i * 8)) & 0xFF;
            }
            let mut scales = [0i32; 8];
            let mut mins = [0i32; 8];
            for j in 0..4usize {
                scales[j] = (sb[j] & 0x3F) as i32;
                mins[j] = (sb[j + 4] & 0x3F) as i32;
            }
            for j in 4..8usize {
                scales[j] = ((sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4)) as i32;
                mins[j] = ((sb[j + 4] >> 4) | ((sb[j] >> 6) << 4)) as i32;
            }

            // q8k.d from activation word 64
            let q8k_d = f32::from_bits(packed_act[act_base + 64]);

            let mut sumi = 0i32;
            let mut summ = 0i32;

            for group in 0..4u32 {
                let is = (group * 2) as usize;
                let qs_base_plane = 4 + group * 8;
                let act_lo_base = (group * 16) as usize;
                let act_hi_base = (group * 16 + 8) as usize;

                let mut isum0 = 0i32;
                let mut isum1 = 0i32;

                for w in 0..8u32 {
                    let qs_plane = blk * 36 + (qs_base_plane + w) as usize;
                    let qs_word = repacked[qs_plane * rows + row];
                    let act_lo = packed_act[act_base + act_lo_base + w as usize];
                    let act_hi = packed_act[act_base + act_hi_base + w as usize];

                    for byte in 0..4u32 {
                        let lo_nib = ((qs_word >> (byte * 8)) & 0x0F) as i32;
                        let hi_nib = ((qs_word >> (byte * 8 + 4)) & 0x0F) as i32;

                        // Q8K signed i8 byte from activation (sign-extend via 24-bit shift)
                        let act_lo_byte = ((act_lo >> (byte * 8)) & 0xFF) as i32;
                        let x_lo = (act_lo_byte << 24) >> 24;
                        let act_hi_byte = ((act_hi >> (byte * 8)) & 0xFF) as i32;
                        let x_hi = (act_hi_byte << 24) >> 24;

                        isum0 += lo_nib * x_lo;
                        isum1 += hi_nib * x_hi;
                    }
                }
                sumi += scales[is] * isum0 + scales[is + 1] * isum1;

                // bsums word: low 16 = bsums[2g], high 16 = bsums[2g+1]
                let bsum_word = packed_act[act_base + 65 + group as usize];
                let bsum_lo = (((bsum_word & 0xFFFF) << 16) as i32) >> 16;
                let bsum_hi = (bsum_word as i32) >> 16;
                summ += mins[is] * bsum_lo + mins[is + 1] * bsum_hi;
            }
            total += q8k_d * (d * sumi as f32 - dmin * summ as f32);
        }
        total
    }

    /// Pure-Rust mirror of `emit_q6k_q8k_gemv` shader logic.
    fn gpu_simulate_q6k_q8k_dot(
        repacked: &[u32],
        row: usize,
        rows: usize,
        packed_act: &[u32],
        blocks_per_row: usize,
    ) -> f32 {
        struct SB {
            ql_off: usize,
            hi: bool,
            qh_off: usize,
            qh_shift: u32,
            elem_off: usize,
            sc_idx: usize,
        }
        let sbs = [
            SB {
                ql_off: 5,
                hi: false,
                qh_off: 37,
                qh_shift: 0,
                elem_off: 0,
                sc_idx: 0,
            },
            SB {
                ql_off: 9,
                hi: false,
                qh_off: 41,
                qh_shift: 0,
                elem_off: 16,
                sc_idx: 1,
            },
            SB {
                ql_off: 13,
                hi: false,
                qh_off: 37,
                qh_shift: 2,
                elem_off: 32,
                sc_idx: 2,
            },
            SB {
                ql_off: 17,
                hi: false,
                qh_off: 41,
                qh_shift: 2,
                elem_off: 48,
                sc_idx: 3,
            },
            SB {
                ql_off: 5,
                hi: true,
                qh_off: 37,
                qh_shift: 4,
                elem_off: 64,
                sc_idx: 4,
            },
            SB {
                ql_off: 9,
                hi: true,
                qh_off: 41,
                qh_shift: 4,
                elem_off: 80,
                sc_idx: 5,
            },
            SB {
                ql_off: 13,
                hi: true,
                qh_off: 37,
                qh_shift: 6,
                elem_off: 96,
                sc_idx: 6,
            },
            SB {
                ql_off: 17,
                hi: true,
                qh_off: 41,
                qh_shift: 6,
                elem_off: 112,
                sc_idx: 7,
            },
            SB {
                ql_off: 21,
                hi: false,
                qh_off: 45,
                qh_shift: 0,
                elem_off: 128,
                sc_idx: 8,
            },
            SB {
                ql_off: 25,
                hi: false,
                qh_off: 49,
                qh_shift: 0,
                elem_off: 144,
                sc_idx: 9,
            },
            SB {
                ql_off: 29,
                hi: false,
                qh_off: 45,
                qh_shift: 2,
                elem_off: 160,
                sc_idx: 10,
            },
            SB {
                ql_off: 33,
                hi: false,
                qh_off: 49,
                qh_shift: 2,
                elem_off: 176,
                sc_idx: 11,
            },
            SB {
                ql_off: 21,
                hi: true,
                qh_off: 45,
                qh_shift: 4,
                elem_off: 192,
                sc_idx: 12,
            },
            SB {
                ql_off: 25,
                hi: true,
                qh_off: 49,
                qh_shift: 4,
                elem_off: 208,
                sc_idx: 13,
            },
            SB {
                ql_off: 29,
                hi: true,
                qh_off: 45,
                qh_shift: 6,
                elem_off: 224,
                sc_idx: 14,
            },
            SB {
                ql_off: 33,
                hi: true,
                qh_off: 49,
                qh_shift: 6,
                elem_off: 240,
                sc_idx: 15,
            },
        ];

        let mut total = 0.0f32;
        for blk in 0..blocks_per_row {
            let plane_base = blk * 53 * rows;
            let act_base = blk * 69;

            // d from plane 0 (low 16 bits)
            let packed = repacked[plane_base + row];
            let d_raw = packed & 0xFFFF;
            let d = gpu_f16_to_f32(d_raw);

            // q8k.d from activation word 64
            let q8k_d = f32::from_bits(packed_act[act_base + 64]);

            // Scales (16 i8) from planes 1..4
            let mut sw = [0u32; 4];
            for i in 0..4 {
                sw[i] = repacked[(blk * 53 + 1 + i) * rows + row];
            }

            let mut sumi = 0i32;
            for sb in &sbs {
                let sw_idx = sb.sc_idx / 4;
                let byte_shift = (sb.sc_idx % 4) * 8;
                let byte_val = ((sw[sw_idx] >> byte_shift) & 0xFF) as u8;
                let scale_i = byte_val as i8 as i32;

                for w in 0..4u32 {
                    let ql_plane = blk * 53 + sb.ql_off + w as usize;
                    let ql_word = repacked[ql_plane * rows + row];
                    let qh_plane = blk * 53 + sb.qh_off + w as usize;
                    let qh_word = repacked[qh_plane * rows + row];

                    for bi in 0..4u32 {
                        let ql_shift = if sb.hi { bi * 8 + 4 } else { bi * 8 };
                        let ql_nibble = (ql_word >> ql_shift) & 0x0F;

                        let qh_shift = bi * 8 + sb.qh_shift;
                        let qh_bits = (qh_word >> qh_shift) & 0x3;

                        let q6 = ql_nibble | (qh_bits << 4);
                        let q6_centered = q6 as i32 - 32;

                        let elem_idx = sb.elem_off + (w * 4 + bi) as usize;
                        let act_word = packed_act[act_base + elem_idx / 4];
                        let act_byte_shift = ((elem_idx % 4) * 8) as u32;
                        let act_byte = ((act_word >> act_byte_shift) & 0xFF) as i32;
                        let q8k_signed = (act_byte << 24) >> 24;

                        sumi += scale_i * q6_centered * q8k_signed;
                    }
                }
            }
            total += d * q8k_d * sumi as f32;
        }
        total
    }

    #[test]
    fn test_q4k_q8k_shader_simulation_matches_cpu() {
        // Build a single Q4_K block with non-trivial values.
        let mut block = [0u8; 144];
        // d = 0.05, dmin = 0.02 (f16 LE)
        block[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
        // 12 scale/min bytes (mix of values to exercise both 6-bit and split-pair paths)
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
        // 128 quant bytes (varied nibble values)
        for i in 0..128 {
            block[16 + i] = ((i * 7 + 11) % 256) as u8;
        }

        // Activation: deterministic varied f32, then quantize to Q8K
        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.41).sin() * 1.7) - 0.3;
        }
        let (qs, q8k_d, bsums) = quantize_q8k_block(&input);
        let packed_act = pack_q8k_for_shader(&qs, q8k_d, &bsums);

        let cpu_ref = cpu_q4k_q8k_integer_dot(&block, &qs, q8k_d, &bsums);

        let repacked = repack_q4k_transposed(&block, 1, 1);
        let gpu_sim = gpu_simulate_q4k_q8k_dot(&repacked, 0, 1, &packed_act, 1);

        eprintln!(
            "Q4_K × Q8K: CPU={cpu_ref}, GPU={gpu_sim}, diff={}",
            (cpu_ref - gpu_sim).abs()
        );
        // Both implementations operate on the same i32 sumi/summ accumulators
        // and apply the same single f32 finalization, so the result must be exact.
        assert_eq!(
            cpu_ref.to_bits(),
            gpu_sim.to_bits(),
            "Q4_K × Q8K shader simulation must match CPU bit-exactly: CPU={cpu_ref}, GPU={gpu_sim}"
        );
    }

    #[test]
    fn test_q6k_q8k_shader_simulation_matches_cpu() {
        // Build a single Q6_K block (210 bytes).
        let mut block = [0u8; 210];
        // ql: 128 varied bytes
        for i in 0..128 {
            block[i] = ((i * 7 + 3) % 256) as u8;
        }
        // qh: 64 varied bytes
        for i in 0..64 {
            block[128 + i] = ((i * 13 + 5) % 256) as u8;
        }
        // 16 signed-i8 scales
        let scales: [i8; 16] = [
            10, -5, 20, -15, 8, -3, 25, -10, 12, -7, 18, -20, 6, -1, 30, -8,
        ];
        for i in 0..16 {
            block[192 + i] = scales[i] as u8;
        }
        // d = 0.05 f16
        block[208..210].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());

        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.37).sin() * 2.0) + 0.5;
        }
        let (qs, q8k_d, bsums) = quantize_q8k_block(&input);
        let packed_act = pack_q8k_for_shader(&qs, q8k_d, &bsums);

        let cpu_ref = cpu_q6k_q8k_integer_dot(&block, &qs, q8k_d);

        let repacked = repack_q6k_transposed(&block, 1, 1);
        let gpu_sim = gpu_simulate_q6k_q8k_dot(&repacked, 0, 1, &packed_act, 1);

        eprintln!(
            "Q6_K × Q8K: CPU={cpu_ref}, GPU={gpu_sim}, diff={}",
            (cpu_ref - gpu_sim).abs()
        );
        assert_eq!(
            cpu_ref.to_bits(),
            gpu_sim.to_bits(),
            "Q6_K × Q8K shader simulation must match CPU bit-exactly: CPU={cpu_ref}, GPU={gpu_sim}"
        );
    }

    /// Pure-Rust mirror of `emit_quantize_to_q8k` shader logic for one block.
    /// Returns 69 u32 packed words matching `pack_q8k_for_shader` semantics.
    fn gpu_simulate_quantize_to_q8k(input: &[f32; 256]) -> Vec<u32> {
        // amax
        let mut amax = 0.0f32;
        for &x in input.iter() {
            amax = amax.max(x.abs());
        }
        let d = amax / 127.0;
        let id_inv = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut output = vec![0u32; 69];
        let mut bsums = [0i32; 8];

        for w in 0..64u32 {
            let group = (w / 8) as usize;
            let mut word = 0u32;
            for byte in 0..4u32 {
                let elem_idx = (w * 4 + byte) as usize;
                let f = input[elem_idx] * id_inv;
                // ties-away-from-zero round mirroring shader's
                // `select(f < 0, -floor(|f|+0.5), floor(|f|+0.5))`
                let abs_f = f.abs();
                let plus_half = abs_f + 0.5;
                let floor_v = plus_half.floor();
                let signed_v = if f < 0.0 { -floor_v } else { floor_v };
                let clamped = signed_v.clamp(-128.0, 127.0);
                let q = clamped as i32; // ConvertFToS, value in [-128, 127]

                bsums[group] += q;
                let q_byte = (q as u32) & 0xFF;
                word |= q_byte << (byte * 8);
            }
            output[w as usize] = word;
        }

        output[64] = d.to_bits();
        for g in 0..4 {
            let lo = (bsums[g * 2] as u32) & 0xFFFF;
            let hi = (bsums[g * 2 + 1] as u32) & 0xFFFF;
            output[65 + g] = lo | (hi << 16);
        }

        output
    }

    #[test]
    fn test_quantize_to_q8k_shader_simulation_matches_cpu() {
        // Deterministic varied input across the full f32 range (positive, negative,
        // small, large, mixed signs).
        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.41).sin() * 1.7) - 0.3 + ((i as f32 * 0.13).cos() * 0.9);
        }

        // CPU reference: same path the rest of the shader simulation tests use
        // (quantize_q8k_block + pack_q8k_for_shader).
        let (qs, d, bsums) = quantize_q8k_block(&input);
        let cpu_packed = pack_q8k_for_shader(&qs, d, &bsums);

        let gpu_packed = gpu_simulate_quantize_to_q8k(&input);

        assert_eq!(cpu_packed.len(), 69);
        assert_eq!(gpu_packed.len(), 69);
        for i in 0..69 {
            assert_eq!(
                cpu_packed[i], gpu_packed[i],
                "Word {i} mismatch: CPU=0x{:08x}, GPU=0x{:08x}",
                cpu_packed[i], gpu_packed[i]
            );
        }
    }

    /// End-to-end: shader-quantized activations should produce the same
    /// `Q4_K × Q8K` integer-dot result as CPU-quantized activations.
    /// This locks in the contract that downstream Q4_K/Q6_K × Q8K shaders
    /// can be fed directly from the on-GPU quantized buffer.
    #[test]
    fn test_quantize_to_q8k_chains_into_q4k_q8k_shader() {
        // Build a Q4_K block.
        let mut block = [0u8; 144];
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
        for i in 0..128 {
            block[16 + i] = ((i * 7 + 11) % 256) as u8;
        }

        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = ((i as f32 * 0.41).sin() * 1.7) - 0.3;
        }

        // Reference: CPU-quantize then run integer-dot mirror.
        let (qs, d, bsums) = quantize_q8k_block(&input);
        let cpu_dot = cpu_q4k_q8k_integer_dot(&block, &qs, d, &bsums);

        // Path under test: GPU-quantize via shader simulation, feed into Q4_K × Q8K simulation.
        let gpu_packed_act = gpu_simulate_quantize_to_q8k(&input);
        let repacked = repack_q4k_transposed(&block, 1, 1);
        let gpu_dot = gpu_simulate_q4k_q8k_dot(&repacked, 0, 1, &gpu_packed_act, 1);

        assert_eq!(
            cpu_dot.to_bits(),
            gpu_dot.to_bits(),
            "End-to-end CPU vs GPU-quantize-then-dot must be bit-exact: CPU={cpu_dot}, GPU={gpu_dot}"
        );
    }
}
