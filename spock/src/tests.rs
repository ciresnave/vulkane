//! Integration tests for the spock Vulkan bindings
//!
//! These tests verify that the generated bindings from vk.xml are correct
//! and complete. They run against the actual generated code included via
//! build.rs.

use crate::raw::bindings::*;

// ============================================================================
// Core type existence tests — verify key Vulkan types were generated
// ============================================================================

#[test]
fn test_handle_types_exist() {
    // Dispatchable handles should be pointer-sized
    let _: VkInstance = std::ptr::null_mut();
    let _: VkDevice = std::ptr::null_mut();
    let _: VkPhysicalDevice = std::ptr::null_mut();
    let _: VkQueue = std::ptr::null_mut();
    let _: VkCommandBuffer = std::ptr::null_mut();
}

#[test]
fn test_non_dispatchable_handle_types_exist() {
    // Non-dispatchable handles should be u64
    let _: VkSemaphore = 0u64;
    let _: VkFence = 0u64;
    let _: VkBuffer = 0u64;
    let _: VkImage = 0u64;
    let _: VkPipeline = 0u64;
    let _: VkSampler = 0u64;
    let _: VkDescriptorSet = 0u64;
    let _: VkFramebuffer = 0u64;
    let _: VkRenderPass = 0u64;
    let _: VkCommandPool = 0u64;
    let _: VkSwapchainKHR = 0u64;
    let _: VkSurfaceKHR = 0u64;
}

#[test]
fn test_base_types_exist() {
    let _: VkBool32 = 0u32;
    let _: VkFlags = 0u32;
    let _: VkFlags64 = 0u64;
    let _: VkDeviceSize = 0u64;
    let _: VkDeviceAddress = 0u64;
    let _: VkSampleMask = 0u32;
}

// ============================================================================
// Struct tests — verify key structs have correct fields
// ============================================================================

#[test]
fn test_vk_application_info_struct() {
    let info = VkApplicationInfo {
        sType: VkStructureType::STRUCTURE_TYPE_APPLICATION_INFO,
        pNext: std::ptr::null(),
        pApplicationName: std::ptr::null(),
        applicationVersion: 0,
        pEngineName: std::ptr::null(),
        engineVersion: 0,
        apiVersion: VK_API_VERSION_1_0,
    };
    assert_eq!(info.apiVersion, VK_API_VERSION_1_0);
}

#[test]
fn test_vk_instance_create_info_struct() {
    let info = VkInstanceCreateInfo {
        sType: VkStructureType::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        pApplicationInfo: std::ptr::null(),
        enabledLayerCount: 0,
        ppEnabledLayerNames: std::ptr::null(),
        enabledExtensionCount: 0,
        ppEnabledExtensionNames: std::ptr::null(),
    };
    assert_eq!(info.enabledLayerCount, 0);
}

#[test]
fn test_struct_default_impls() {
    let app_info = VkApplicationInfo::default();
    assert!(app_info.pNext.is_null());
    assert!(app_info.pApplicationName.is_null());

    let create_info = VkInstanceCreateInfo::default();
    assert!(create_info.pNext.is_null());
    assert_eq!(create_info.enabledLayerCount, 0);
}

// ============================================================================
// Enum tests — verify enums have correct variants including extensions
// ============================================================================

#[test]
fn test_vk_result_enum() {
    // Core values
    assert_eq!(VkResult::SUCCESS as i32, 0);
    assert_eq!(VkResult::NOT_READY as i32, 1);
    assert_eq!(VkResult::TIMEOUT as i32, 2);
    assert_eq!(VkResult::ERROR_OUT_OF_HOST_MEMORY as i32, -1);
    assert_eq!(VkResult::ERROR_OUT_OF_DEVICE_MEMORY as i32, -2);
    assert_eq!(VkResult::ERROR_DEVICE_LOST as i32, -4);
}

#[test]
fn test_vk_structure_type_has_extension_values() {
    // Core value
    assert_eq!(VkStructureType::STRUCTURE_TYPE_APPLICATION_INFO as i32, 0);
    // Extension value (VK_KHR_swapchain, extnumber=2, offset=0)
    // Value = 1000000000 + (2-1)*1000 + 0 = 1000001000
    assert_eq!(
        VkStructureType::STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR as i32,
        1000001000
    );
}

// ============================================================================
// Version macro tests — verify functions derived from vk.xml work correctly
// ============================================================================

#[test]
fn test_vk_make_api_version() {
    let version = vk_make_api_version(0, 1, 0, 0);
    assert_eq!(version, VK_API_VERSION_1_0);
}

#[test]
fn test_vk_api_version_extraction() {
    // Use VK_API_VERSION_1_0 which exists in all supported Vulkan versions
    let version = VK_API_VERSION_1_0;
    assert_eq!(vk_api_version_variant(version), 0);
    assert_eq!(vk_api_version_major(version), 1);
    assert_eq!(vk_api_version_minor(version), 0);
    assert_eq!(vk_api_version_patch(version), 0);
}

#[test]
fn test_vk_header_version_exists() {
    // VK_HEADER_VERSION should be parsed from vk.xml, not hardcoded
    assert!(VK_HEADER_VERSION > 0);
}

#[test]
fn test_version_roundtrip() {
    // Encoding then decoding should produce the same values
    let v = vk_make_api_version(0, 1, 4, 42);
    assert_eq!(vk_api_version_variant(v), 0);
    assert_eq!(vk_api_version_major(v), 1);
    assert_eq!(vk_api_version_minor(v), 4);
    assert_eq!(vk_api_version_patch(v), 42);
}

// ============================================================================
// Constants tests
// ============================================================================

#[test]
fn test_constants_exist() {
    assert!(VK_MAX_PHYSICAL_DEVICE_NAME_SIZE > 0);
    assert!(VK_MAX_EXTENSION_NAME_SIZE > 0);
    assert!(VK_UUID_SIZE > 0);
    assert!(VK_MAX_MEMORY_TYPES > 0);
    assert!(VK_MAX_MEMORY_HEAPS > 0);
}

// ============================================================================
// Function pointer typedef tests
// ============================================================================

#[test]
fn test_function_pointer_types_exist() {
    // These are type aliases for unsafe extern "system" fn(...)
    // We can't instantiate them but we can verify they're valid types
    let _: Option<vkCreateInstance> = None;
    let _: Option<vkDestroyInstance> = None;
    let _: Option<vkEnumeratePhysicalDevices> = None;
    let _: Option<vkCreateDevice> = None;
    let _: Option<vkDestroyDevice> = None;
    let _: Option<vkGetDeviceProcAddr> = None;
    let _: Option<vkGetInstanceProcAddr> = None;
}

#[test]
fn test_function_pointer_signatures_match_spec() {
    // Verify that generated function pointer typedefs have the exact
    // signatures specified in the Vulkan API. We do this by assigning
    // stub functions with the expected signatures to the typedefs.
    // This catches regressions in parameter ordering, parameter types,
    // calling convention, and return type.

    // vkCreateInstance: VkResult vkCreateInstance(
    //     const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance*)
    unsafe extern "system" fn create_instance_stub(
        _create_info: *const VkInstanceCreateInfo,
        _allocator: *const VkAllocationCallbacks,
        _instance: *mut VkInstance,
    ) -> VkResult {
        VkResult::SUCCESS
    }
    let _f: vkCreateInstance = create_instance_stub;

    // vkDestroyInstance: void vkDestroyInstance(VkInstance, const VkAllocationCallbacks*)
    unsafe extern "system" fn destroy_instance_stub(
        _instance: VkInstance,
        _allocator: *const VkAllocationCallbacks,
    ) {
    }
    let _f: vkDestroyInstance = destroy_instance_stub;

    // vkEnumeratePhysicalDevices
    unsafe extern "system" fn enum_phys_stub(
        _instance: VkInstance,
        _count: *mut u32,
        _devices: *mut VkPhysicalDevice,
    ) -> VkResult {
        VkResult::SUCCESS
    }
    let _f: vkEnumeratePhysicalDevices = enum_phys_stub;

    // vkEnumerateInstanceVersion: VkResult vkEnumerateInstanceVersion(uint32_t*)
    unsafe extern "system" fn enum_version_stub(_version: *mut u32) -> VkResult {
        VkResult::SUCCESS
    }
    let _f: vkEnumerateInstanceVersion = enum_version_stub;
}

#[test]
fn test_dispatch_tables_have_required_fields() {
    // VkEntryDispatchTable must contain the global Vulkan functions
    let entry: VkEntryDispatchTable = unsafe { std::mem::zeroed() };
    let _ = entry.vkCreateInstance;
    let _ = entry.vkEnumerateInstanceLayerProperties;
    let _ = entry.vkEnumerateInstanceExtensionProperties;
    let _ = entry.vkEnumerateInstanceVersion;

    // VkInstanceDispatchTable must contain instance-level functions
    let inst: VkInstanceDispatchTable = unsafe { std::mem::zeroed() };
    let _ = inst.vkDestroyInstance;
    let _ = inst.vkEnumeratePhysicalDevices;
    let _ = inst.vkGetPhysicalDeviceProperties;
    let _ = inst.vkGetPhysicalDeviceMemoryProperties;
    let _ = inst.vkGetPhysicalDeviceQueueFamilyProperties;
    let _ = inst.vkCreateDevice;

    // VkDeviceDispatchTable must contain device-level functions
    let dev: VkDeviceDispatchTable = unsafe { std::mem::zeroed() };
    let _ = dev.vkDestroyDevice;
    let _ = dev.vkGetDeviceQueue;
    let _ = dev.vkCreateBuffer;
    let _ = dev.vkAllocateMemory;
    let _ = dev.vkCreateCommandPool;
}

// ============================================================================
// Dispatch table tests
// ============================================================================

#[test]
fn test_dispatch_tables_exist() {
    // Verify the generated dispatch table structs exist as types
    let _: Option<Box<VkEntryDispatchTable>> = None;
    let _: Option<Box<VkInstanceDispatchTable>> = None;
    let _: Option<Box<VkDeviceDispatchTable>> = None;
}

// ============================================================================
// Extension struct tests — verify extension-defined structs are generated
// ============================================================================

#[test]
fn test_extension_structs_exist() {
    // VK_KHR_swapchain
    let _info = VkSwapchainCreateInfoKHR::default();

    // VK_EXT_debug_utils
    let _info = VkDebugUtilsMessengerCreateInfoEXT::default();
}

// ============================================================================
// Union type tests
// ============================================================================

#[test]
fn test_clear_color_value_is_union() {
    // VkClearColorValue must be a union — all fields overlap in memory
    assert_eq!(
        std::mem::size_of::<VkClearColorValue>(),
        std::mem::size_of::<[f32; 4]>(),
        "VkClearColorValue should be the size of its largest field, not a sum"
    );

    let mut color = VkClearColorValue::default();
    unsafe {
        color.float32 = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(color.float32[0], 1.0);
    }
}

#[test]
fn test_clear_value_is_union() {
    // VkClearValue contains VkClearColorValue (a union) — should itself be a union
    let val = VkClearValue::default();
    // Size should be max of its fields, not sum
    assert!(
        std::mem::size_of::<VkClearValue>() <= 16,
        "VkClearValue should be at most 16 bytes (4 x f32), got {}",
        std::mem::size_of::<VkClearValue>()
    );
    let _ = val;
}

#[test]
fn test_null_handle() {
    // VK_NULL_HANDLE should be 0, derived from vk.xml
    assert_eq!(VK_NULL_HANDLE, 0u64);
}

// ============================================================================
// Struct layout assertions
// ============================================================================

#[test]
fn test_vk_application_info_size_matches_spec() {
    // VkApplicationInfo per the Vulkan spec is:
    //   VkStructureType    sType;             // 4 bytes (i32)
    //   const void*        pNext;             // 8 bytes on 64-bit
    //   const char*        pApplicationName;  // 8
    //   uint32_t           applicationVersion;// 4
    //   const char*        pEngineName;       // 8
    //   uint32_t           engineVersion;     // 4
    //   uint32_t           apiVersion;        // 4
    // With C alignment, this is 48 bytes on 64-bit platforms.
    #[cfg(target_pointer_width = "64")]
    assert_eq!(std::mem::size_of::<VkApplicationInfo>(), 48);

    // Independent of pointer width, alignment must be at least 8
    // (it contains pointers).
    #[cfg(target_pointer_width = "64")]
    assert_eq!(std::mem::align_of::<VkApplicationInfo>(), 8);
}

#[test]
fn test_vk_extent_3d_layout() {
    // VkExtent3D is { uint32_t width, height, depth } — 12 bytes, aligned to 4
    assert_eq!(std::mem::size_of::<VkExtent3D>(), 12);
    assert_eq!(std::mem::align_of::<VkExtent3D>(), 4);

    let e = VkExtent3D {
        width: 1024,
        height: 768,
        depth: 1,
    };
    assert_eq!(e.width, 1024);
    assert_eq!(e.height, 768);
    assert_eq!(e.depth, 1);
}

#[test]
fn test_vk_offset_2d_layout() {
    // VkOffset2D is { int32_t x, y } — 8 bytes
    assert_eq!(std::mem::size_of::<VkOffset2D>(), 8);
    assert_eq!(std::mem::align_of::<VkOffset2D>(), 4);
}

#[test]
fn test_vk_rect_2d_layout() {
    // VkRect2D is { VkOffset2D offset; VkExtent2D extent; }
    // offset = 8 bytes, extent (uint32_t x 2) = 8 bytes, total 16
    assert_eq!(std::mem::size_of::<VkRect2D>(), 16);
}

// ============================================================================
// VkResult helper tests
// ============================================================================

#[test]
fn test_vk_result_into_result_success() {
    use crate::raw::VkResultExt;
    assert!(VkResult::SUCCESS.into_result().is_ok());
}

#[test]
fn test_vk_result_into_result_error() {
    use crate::raw::VkResultExt;
    let err = VkResult::ERROR_OUT_OF_HOST_MEMORY.into_result();
    assert!(err.is_err());
    assert_eq!(err.unwrap_err(), VkResult::ERROR_OUT_OF_HOST_MEMORY);
}

#[test]
fn test_vk_result_is_success_and_is_error() {
    use crate::raw::VkResultExt;
    assert!(VkResult::SUCCESS.is_success());
    assert!(!VkResult::SUCCESS.is_error());

    assert!(!VkResult::ERROR_DEVICE_LOST.is_success());
    assert!(VkResult::ERROR_DEVICE_LOST.is_error());

    // NOT_READY is positive (1), so it's not "success" but also not "error"
    assert!(!VkResult::NOT_READY.is_success());
    assert!(!VkResult::NOT_READY.is_error());
}

// ============================================================================
// Bitmask flag tests
// ============================================================================

#[test]
fn test_bitmask_flags_combinable() {
    // Bitmask flags should be emitted as pub const so they can be combined with |
    let flags: VkBufferUsageFlagBits =
        BUFFER_USAGE_TRANSFER_SRC_BIT | BUFFER_USAGE_TRANSFER_DST_BIT;
    assert!(flags != 0);
}
