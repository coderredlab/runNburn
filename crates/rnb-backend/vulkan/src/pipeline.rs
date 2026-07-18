use crate::context::{GpuBuffer, VulkanContext};
use crate::ffi::types::*;
use std::ptr;

/// Vulkan compute pipeline + multiple descriptor sets for batch dispatch.
/// Command pool/buffer/fence are managed externally by VulkanLayerGemv.
pub(crate) struct ComputePipeline {
    pub(crate) descriptor_set_layout: VkDescriptorSetLayout,
    pub(crate) pipeline_layout: VkPipelineLayout,
    pub(crate) pipeline: VkPipeline,
    pub(crate) descriptor_pool: VkDescriptorPool,
    pub(crate) descriptor_sets: Vec<VkDescriptorSet>,
}

impl ComputePipeline {
    /// Create a compute pipeline with 2 storage buffer bindings + push constants.
    pub(crate) unsafe fn new_2binding(
        ctx: &VulkanContext,
        spirv: &[u32],
        max_sets: u32,
        push_size: u32,
    ) -> Result<Self, String> {
        Self::new_nbinding(ctx, spirv, max_sets, 2, push_size)
    }

    /// Create a compute pipeline with N storage buffer bindings + push constants.
    pub(crate) unsafe fn new_nbinding(
        ctx: &VulkanContext,
        spirv: &[u32],
        max_sets: u32,
        num_bindings: u32,
        push_size: u32,
    ) -> Result<Self, String> {
        let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..num_bindings)
            .map(|i| VkDescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            })
            .collect();

        let dsl_create_info = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            binding_count: num_bindings,
            p_bindings: bindings.as_ptr(),
        };
        let mut descriptor_set_layout: VkDescriptorSetLayout = VK_NULL_HANDLE;
        let res = (ctx.vk.create_descriptor_set_layout)(
            ctx.device,
            &dsl_create_info,
            ptr::null(),
            &mut descriptor_set_layout,
        );
        if res != VK_SUCCESS {
            return Err(format!("vkCreateDescriptorSetLayout failed: {}", res));
        }

        let push_range = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: push_size,
        };
        let pl_create_info = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
        };
        let mut pipeline_layout: VkPipelineLayout = VK_NULL_HANDLE;
        let res = (ctx.vk.create_pipeline_layout)(
            ctx.device,
            &pl_create_info,
            ptr::null(),
            &mut pipeline_layout,
        );
        if res != VK_SUCCESS {
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkCreatePipelineLayout failed: {}", res));
        }

        let sm_create_info = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            code_size: spirv.len() * 4,
            p_code: spirv.as_ptr(),
        };
        let mut shader_module: VkShaderModule = VK_NULL_HANDLE;
        let res = (ctx.vk.create_shader_module)(
            ctx.device,
            &sm_create_info,
            ptr::null(),
            &mut shader_module,
        );
        if res != VK_SUCCESS {
            (ctx.vk.destroy_pipeline_layout)(ctx.device, pipeline_layout, ptr::null());
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkCreateShaderModule failed: {}", res));
        }

        let stage = VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            p_name: b"main\0".as_ptr(),
            p_specialization_info: ptr::null(),
        };
        let cp_create_info = VkComputePipelineCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: VK_NULL_HANDLE,
            base_pipeline_index: -1,
        };
        let mut pipeline: VkPipeline = VK_NULL_HANDLE;
        let res = (ctx.vk.create_compute_pipelines)(
            ctx.device,
            VK_NULL_HANDLE,
            1,
            &cp_create_info,
            ptr::null(),
            &mut pipeline,
        );
        (ctx.vk.destroy_shader_module)(ctx.device, shader_module, ptr::null());
        if res != VK_SUCCESS {
            (ctx.vk.destroy_pipeline_layout)(ctx.device, pipeline_layout, ptr::null());
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkCreateComputePipelines failed: {}", res));
        }

        let pool_size = VkDescriptorPoolSize {
            typ: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: num_bindings * max_sets,
        };
        let dp_create_info = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            max_sets,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut descriptor_pool: VkDescriptorPool = VK_NULL_HANDLE;
        let res = (ctx.vk.create_descriptor_pool)(
            ctx.device,
            &dp_create_info,
            ptr::null(),
            &mut descriptor_pool,
        );
        if res != VK_SUCCESS {
            (ctx.vk.destroy_pipeline)(ctx.device, pipeline, ptr::null());
            (ctx.vk.destroy_pipeline_layout)(ctx.device, pipeline_layout, ptr::null());
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkCreateDescriptorPool failed: {}", res));
        }

        let layouts = vec![descriptor_set_layout; max_sets as usize];
        let ds_alloc_info = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool,
            descriptor_set_count: max_sets,
            p_set_layouts: layouts.as_ptr(),
        };
        let mut descriptor_sets = vec![0u64; max_sets as usize];
        let res = (ctx.vk.allocate_descriptor_sets)(
            ctx.device,
            &ds_alloc_info,
            descriptor_sets.as_mut_ptr(),
        );
        if res != VK_SUCCESS {
            (ctx.vk.destroy_descriptor_pool)(ctx.device, descriptor_pool, ptr::null());
            (ctx.vk.destroy_pipeline)(ctx.device, pipeline, ptr::null());
            (ctx.vk.destroy_pipeline_layout)(ctx.device, pipeline_layout, ptr::null());
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkAllocateDescriptorSets failed: {}", res));
        }

        Ok(Self {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_sets,
        })
    }

    /// Create a compute pipeline from pre-assembled SPIR-V words.
    /// `max_sets`: number of descriptor sets to allocate (for batch dispatch).
    pub(crate) unsafe fn new(
        ctx: &VulkanContext,
        spirv: &[u32],
        max_sets: u32,
    ) -> Result<Self, String> {
        // ------------------------------------------------------------------
        // 1. Descriptor set layout — 3 storage buffer bindings (weight, input, output)
        // ------------------------------------------------------------------
        let bindings = [
            VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
        ];

        let dsl_create_info = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            binding_count: 3,
            p_bindings: bindings.as_ptr(),
        };

        let mut descriptor_set_layout: VkDescriptorSetLayout = VK_NULL_HANDLE;
        let res = (ctx.vk.create_descriptor_set_layout)(
            ctx.device,
            &dsl_create_info,
            ptr::null(),
            &mut descriptor_set_layout,
        );
        if res != VK_SUCCESS {
            return Err(format!("vkCreateDescriptorSetLayout failed: {}", res));
        }

        // ------------------------------------------------------------------
        // 2. Pipeline layout — descriptor set + push constant range (12 bytes)
        // ------------------------------------------------------------------
        let push_range = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: 12, // 3 × u32: rows, cols, rows_per_wg
        };

        let pl_create_info = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
        };

        let mut pipeline_layout: VkPipelineLayout = VK_NULL_HANDLE;
        let res = (ctx.vk.create_pipeline_layout)(
            ctx.device,
            &pl_create_info,
            ptr::null(),
            &mut pipeline_layout,
        );
        if res != VK_SUCCESS {
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkCreatePipelineLayout failed: {}", res));
        }

        // ------------------------------------------------------------------
        // 3. Shader module
        // ------------------------------------------------------------------
        let sm_create_info = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            code_size: spirv.len() * 4,
            p_code: spirv.as_ptr(),
        };

        let mut shader_module: VkShaderModule = VK_NULL_HANDLE;
        let res = (ctx.vk.create_shader_module)(
            ctx.device,
            &sm_create_info,
            ptr::null(),
            &mut shader_module,
        );
        if res != VK_SUCCESS {
            (ctx.vk.destroy_pipeline_layout)(ctx.device, pipeline_layout, ptr::null());
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkCreateShaderModule failed: {}", res));
        }

        // ------------------------------------------------------------------
        // 4. Compute pipeline
        // ------------------------------------------------------------------
        let stage = VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            p_name: b"main\0".as_ptr(),
            p_specialization_info: ptr::null(),
        };

        let cp_create_info = VkComputePipelineCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: VK_NULL_HANDLE,
            base_pipeline_index: -1,
        };

        let mut pipeline: VkPipeline = VK_NULL_HANDLE;
        let res = (ctx.vk.create_compute_pipelines)(
            ctx.device,
            VK_NULL_HANDLE,
            1,
            &cp_create_info,
            ptr::null(),
            &mut pipeline,
        );
        // Shader module no longer needed after pipeline creation
        (ctx.vk.destroy_shader_module)(ctx.device, shader_module, ptr::null());
        if res != VK_SUCCESS {
            (ctx.vk.destroy_pipeline_layout)(ctx.device, pipeline_layout, ptr::null());
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkCreateComputePipelines failed: {}", res));
        }

        // ------------------------------------------------------------------
        // 5. Descriptor pool + multiple descriptor sets for batch dispatch
        // ------------------------------------------------------------------
        let pool_size = VkDescriptorPoolSize {
            typ: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 3 * max_sets,
        };

        let dp_create_info = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            max_sets,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };

        let mut descriptor_pool: VkDescriptorPool = VK_NULL_HANDLE;
        let res = (ctx.vk.create_descriptor_pool)(
            ctx.device,
            &dp_create_info,
            ptr::null(),
            &mut descriptor_pool,
        );
        if res != VK_SUCCESS {
            (ctx.vk.destroy_pipeline)(ctx.device, pipeline, ptr::null());
            (ctx.vk.destroy_pipeline_layout)(ctx.device, pipeline_layout, ptr::null());
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkCreateDescriptorPool failed: {}", res));
        }

        // Allocate max_sets descriptor sets, all with the same layout
        let layouts = vec![descriptor_set_layout; max_sets as usize];
        let ds_alloc_info = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool,
            descriptor_set_count: max_sets,
            p_set_layouts: layouts.as_ptr(),
        };

        let mut descriptor_sets = vec![0u64; max_sets as usize];
        let res = (ctx.vk.allocate_descriptor_sets)(
            ctx.device,
            &ds_alloc_info,
            descriptor_sets.as_mut_ptr(),
        );
        if res != VK_SUCCESS {
            (ctx.vk.destroy_descriptor_pool)(ctx.device, descriptor_pool, ptr::null());
            (ctx.vk.destroy_pipeline)(ctx.device, pipeline, ptr::null());
            (ctx.vk.destroy_pipeline_layout)(ctx.device, pipeline_layout, ptr::null());
            (ctx.vk.destroy_descriptor_set_layout)(ctx.device, descriptor_set_layout, ptr::null());
            return Err(format!("vkAllocateDescriptorSets failed: {}", res));
        }

        Ok(Self {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_sets,
        })
    }

    /// Update the 3 descriptor set bindings for a specific set index.
    pub(crate) unsafe fn bind_buffers(
        &self,
        ctx: &VulkanContext,
        set_idx: usize,
        weight_buf: &GpuBuffer,
        weight_size: u64,
        input_buf: &GpuBuffer,
        input_size: u64,
        output_buf: &GpuBuffer,
        output_size: u64,
    ) {
        let ds = self.descriptor_sets[set_idx];
        let buf_infos = [
            VkDescriptorBufferInfo {
                buffer: weight_buf.buffer,
                offset: 0,
                range: weight_size,
            },
            VkDescriptorBufferInfo {
                buffer: input_buf.buffer,
                offset: 0,
                range: input_size,
            },
            VkDescriptorBufferInfo {
                buffer: output_buf.buffer,
                offset: 0,
                range: output_size,
            },
        ];

        let writes = [
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[0],
                p_texel_buffer_view: ptr::null(),
            },
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[1],
                p_texel_buffer_view: ptr::null(),
            },
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[2],
                p_texel_buffer_view: ptr::null(),
            },
        ];

        (ctx.vk.update_descriptor_sets)(ctx.device, 3, writes.as_ptr(), 0, ptr::null());
    }

    pub(crate) unsafe fn bind_buffers_with_offsets(
        &self,
        ctx: &VulkanContext,
        set_idx: usize,
        weight_buf: &GpuBuffer,
        weight_offset: u64,
        weight_size: u64,
        input_buf: &GpuBuffer,
        input_offset: u64,
        input_size: u64,
        output_buf: &GpuBuffer,
        output_offset: u64,
        output_size: u64,
    ) {
        let ds = self.descriptor_sets[set_idx];
        let buf_infos = [
            VkDescriptorBufferInfo {
                buffer: weight_buf.buffer,
                offset: weight_offset,
                range: weight_size,
            },
            VkDescriptorBufferInfo {
                buffer: input_buf.buffer,
                offset: input_offset,
                range: input_size,
            },
            VkDescriptorBufferInfo {
                buffer: output_buf.buffer,
                offset: output_offset,
                range: output_size,
            },
        ];

        let writes = [
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[0],
                p_texel_buffer_view: ptr::null(),
            },
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[1],
                p_texel_buffer_view: ptr::null(),
            },
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[2],
                p_texel_buffer_view: ptr::null(),
            },
        ];

        (ctx.vk.update_descriptor_sets)(ctx.device, 3, writes.as_ptr(), 0, ptr::null());
    }

    pub(crate) unsafe fn bind_n_buffers_with_offsets(
        &self,
        ctx: &VulkanContext,
        set_idx: usize,
        buffers: &[(&GpuBuffer, u64, u64)],
    ) {
        let ds = self.descriptor_sets[set_idx];
        let buf_infos: Vec<VkDescriptorBufferInfo> = buffers
            .iter()
            .map(|(buf, offset, size)| VkDescriptorBufferInfo {
                buffer: buf.buffer,
                offset: *offset,
                range: *size,
            })
            .collect();
        let writes: Vec<VkWriteDescriptorSet> = (0..buf_infos.len())
            .map(|i| VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: i as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[i],
                p_texel_buffer_view: ptr::null(),
            })
            .collect();

        (ctx.vk.update_descriptor_sets)(
            ctx.device,
            writes.len() as u32,
            writes.as_ptr(),
            0,
            ptr::null(),
        );
    }

    /// Update only the input buffer (binding 1) for a specific set.
    pub(crate) unsafe fn bind_input(
        &self,
        ctx: &VulkanContext,
        set_idx: usize,
        buf: &GpuBuffer,
        size: u64,
    ) {
        let ds = self.descriptor_sets[set_idx];
        let buf_info = VkDescriptorBufferInfo {
            buffer: buf.buffer,
            offset: 0,
            range: size,
        };
        let write = VkWriteDescriptorSet {
            s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: ds,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buf_info,
            p_texel_buffer_view: ptr::null(),
        };
        (ctx.vk.update_descriptor_sets)(ctx.device, 1, &write, 0, ptr::null());
    }

    /// Update only the output buffer (binding 2) for a specific set.
    pub(crate) unsafe fn bind_output(
        &self,
        ctx: &VulkanContext,
        set_idx: usize,
        buf: &GpuBuffer,
        size: u64,
    ) {
        let ds = self.descriptor_sets[set_idx];
        let buf_info = VkDescriptorBufferInfo {
            buffer: buf.buffer,
            offset: 0,
            range: size,
        };
        let write = VkWriteDescriptorSet {
            s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: ds,
            dst_binding: 2,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buf_info,
            p_texel_buffer_view: ptr::null(),
        };
        (ctx.vk.update_descriptor_sets)(ctx.device, 1, &write, 0, ptr::null());
    }

    /// Update 2 descriptor set bindings (for elementwise shaders).
    pub(crate) unsafe fn bind_buffers_2(
        &self,
        ctx: &VulkanContext,
        set_idx: usize,
        buf_a: &GpuBuffer,
        size_a: u64,
        buf_b: &GpuBuffer,
        size_b: u64,
    ) {
        let ds = self.descriptor_sets[set_idx];
        let buf_infos = [
            VkDescriptorBufferInfo {
                buffer: buf_a.buffer,
                offset: 0,
                range: size_a,
            },
            VkDescriptorBufferInfo {
                buffer: buf_b.buffer,
                offset: 0,
                range: size_b,
            },
        ];
        let writes = [
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[0],
                p_texel_buffer_view: ptr::null(),
            },
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[1],
                p_texel_buffer_view: ptr::null(),
            },
        ];
        (ctx.vk.update_descriptor_sets)(ctx.device, 2, writes.as_ptr(), 0, ptr::null());
    }

    pub(crate) unsafe fn bind_buffers_2_with_offsets(
        &self,
        ctx: &VulkanContext,
        set_idx: usize,
        buf_a: &GpuBuffer,
        offset_a: u64,
        size_a: u64,
        buf_b: &GpuBuffer,
        offset_b: u64,
        size_b: u64,
    ) {
        let ds = self.descriptor_sets[set_idx];
        let buf_infos = [
            VkDescriptorBufferInfo {
                buffer: buf_a.buffer,
                offset: offset_a,
                range: size_a,
            },
            VkDescriptorBufferInfo {
                buffer: buf_b.buffer,
                offset: offset_b,
                range: size_b,
            },
        ];
        let writes = [
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[0],
                p_texel_buffer_view: ptr::null(),
            },
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: ds,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buf_infos[1],
                p_texel_buffer_view: ptr::null(),
            },
        ];
        (ctx.vk.update_descriptor_sets)(ctx.device, 2, writes.as_ptr(), 0, ptr::null());
    }

    /// Destroy all Vulkan objects owned by this pipeline.
    pub(crate) unsafe fn destroy(self, ctx: &VulkanContext) {
        (ctx.vk.destroy_descriptor_pool)(ctx.device, self.descriptor_pool, ptr::null());
        (ctx.vk.destroy_pipeline)(ctx.device, self.pipeline, ptr::null());
        (ctx.vk.destroy_pipeline_layout)(ctx.device, self.pipeline_layout, ptr::null());
        (ctx.vk.destroy_descriptor_set_layout)(ctx.device, self.descriptor_set_layout, ptr::null());
    }
}
