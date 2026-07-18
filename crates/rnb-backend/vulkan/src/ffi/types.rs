#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;

// Dispatchable handles (pointers)
pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;

// Non-dispatchable handles (u64 on 64-bit)
pub type VkDeviceMemory = u64;
pub type VkBuffer = u64;
pub type VkShaderModule = u64;
pub type VkDescriptorSetLayout = u64;
pub type VkPipelineLayout = u64;
pub type VkPipeline = u64;
pub type VkDescriptorPool = u64;
pub type VkDescriptorSet = u64;
pub type VkCommandPool = u64;
pub type VkFence = u64;

pub type VkDeviceSize = u64;
pub type VkFlags = u32;
pub type VkBool32 = u32;
pub type VkResult = i32;
pub type VkStructureType = i32;

pub const VK_NULL_HANDLE: u64 = 0;

// VkResult values
pub const VK_SUCCESS: VkResult = 0;
pub const VK_NOT_READY: VkResult = 1;
pub const VK_TIMEOUT: VkResult = 2;
pub const VK_ERROR_OUT_OF_HOST_MEMORY: VkResult = -1;
pub const VK_ERROR_OUT_OF_DEVICE_MEMORY: VkResult = -2;
pub const VK_ERROR_INITIALIZATION_FAILED: VkResult = -3;
pub const VK_ERROR_DEVICE_LOST: VkResult = -4;

// VkStructureType values (compute shader에 필요한 것만)
pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: VkStructureType = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: VkStructureType = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: VkStructureType = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: VkStructureType = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: VkStructureType = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: VkStructureType = 5;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: VkStructureType = 8;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: VkStructureType = 12;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: VkStructureType = 16;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: VkStructureType = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: VkStructureType = 30;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: VkStructureType = 18;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: VkStructureType = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: VkStructureType = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: VkStructureType = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: VkStructureType = 35;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: VkStructureType = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: VkStructureType = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: VkStructureType = 42;
pub const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER: VkStructureType = 44;

// VkBufferUsageFlags
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkFlags = 0x00000020;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkFlags = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkFlags = 0x00000002;

// VkMemoryPropertyFlags
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkFlags = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkFlags = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkFlags = 0x00000004;

// VkMemoryHeapFlags
pub const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: VkFlags = 0x00000001;

// VkSharingMode
pub const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

// VkQueueFlags
pub const VK_QUEUE_COMPUTE_BIT: VkFlags = 0x00000002;

// VkDescriptorType
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;

// VkShaderStageFlags
pub const VK_SHADER_STAGE_COMPUTE_BIT: VkFlags = 0x00000020;

// VkCommandBufferLevel
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;

// VkCommandBufferUsageFlags
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkFlags = 0x00000001;
pub const VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: VkFlags = 0x00000004;

// VkFenceCreateFlags
pub const VK_FENCE_CREATE_SIGNALED_BIT: VkFlags = 0x00000001;

// VkPipelineBindPoint
pub const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;

// VkPipelineStageFlags (for barriers)
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: VkFlags = 0x00000800;
pub const VK_PIPELINE_STAGE_TRANSFER_BIT: VkFlags = 0x00001000;
pub const VK_PIPELINE_STAGE_HOST_BIT: VkFlags = 0x00004000;

// VkAccessFlags
pub const VK_ACCESS_SHADER_WRITE_BIT: VkFlags = 0x00000040;
pub const VK_ACCESS_SHADER_READ_BIT: VkFlags = 0x00000020;
pub const VK_ACCESS_HOST_READ_BIT: VkFlags = 0x00002000;
pub const VK_ACCESS_TRANSFER_READ_BIT: VkFlags = 0x00000800;
pub const VK_ACCESS_TRANSFER_WRITE_BIT: VkFlags = 0x00001000;

// Misc
pub const VK_API_VERSION_1_1: u32 = (1 << 22) | (1 << 12);
pub const VK_WHOLE_SIZE: VkDeviceSize = !0u64;
pub const VK_QUEUE_FAMILY_IGNORED: u32 = !0u32;

#[repr(C)]
pub struct VkApplicationInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub p_application_name: *const u8,
    pub application_version: u32,
    pub p_engine_name: *const u8,
    pub engine_version: u32,
    pub api_version: u32,
}

#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub p_application_info: *const VkApplicationInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const u8,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const u8,
}

#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub p_queue_priorities: *const f32,
}

#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_create_info_count: u32,
    pub p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const u8,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const u8,
    pub p_enabled_features: *const c_void,
}

#[repr(C)]
pub struct VkQueueFamilyProperties {
    pub queue_flags: VkFlags,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: [u32; 3],
}

#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memory_type_count: u32,
    pub memory_types: [VkMemoryType; 32],
    pub memory_heap_count: u32,
    pub memory_heaps: [VkMemoryHeap; 16],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkMemoryType {
    pub property_flags: VkFlags,
    pub heap_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: VkFlags,
}

#[repr(C)]
pub struct VkBufferCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub size: VkDeviceSize,
    pub usage: VkFlags,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

#[repr(C)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memory_type_bits: u32,
}

#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub allocation_size: VkDeviceSize,
    pub memory_type_index: u32,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: VkFlags,
    pub p_immutable_samplers: *const c_void,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub binding_count: u32,
    pub p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub set_layout_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
pub struct VkPushConstantRange {
    pub stage_flags: VkFlags,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub code_size: usize,
    pub p_code: *const u32,
}

#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub stage: VkFlags,
    pub module: VkShaderModule,
    pub p_name: *const u8,
    pub p_specialization_info: *const c_void,
}

#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub base_pipeline_handle: VkPipeline,
    pub base_pipeline_index: i32,
}

#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub typ: u32,
    pub descriptor_count: u32,
}

#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub descriptor_pool: VkDescriptorPool,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub dst_set: VkDescriptorSet,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: u32,
    pub p_image_info: *const c_void,
    pub p_buffer_info: *const VkDescriptorBufferInfo,
    pub p_texel_buffer_view: *const c_void,
}

#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_family_index: u32,
}

#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub command_pool: VkCommandPool,
    pub level: u32,
    pub command_buffer_count: u32,
}

#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub p_inheritance_info: *const c_void,
}

#[repr(C)]
pub struct VkSubmitInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const c_void,
    pub p_wait_dst_stage_mask: *const VkFlags,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const VkCommandBuffer,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const c_void,
}

#[repr(C)]
pub struct VkFenceCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkFlags,
}

#[repr(C)]
pub struct VkBufferMemoryBarrier {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub src_access_mask: VkFlags,
    pub dst_access_mask: VkFlags,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[repr(C)]
pub struct VkBufferCopy {
    pub src_offset: VkDeviceSize,
    pub dst_offset: VkDeviceSize,
    pub size: VkDeviceSize,
}
