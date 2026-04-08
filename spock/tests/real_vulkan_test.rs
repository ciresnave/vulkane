//! Integration test that exercises the loader against a real Vulkan driver.
//!
//! This test:
//! 1. Loads the Vulkan shared library (vulkan-1.dll / libvulkan.so.1)
//! 2. Loads the entry-level dispatch table
//! 3. Calls vkEnumerateInstanceVersion to verify dispatch works
//! 4. Creates a real VkInstance via vkCreateInstance
//! 5. Loads the instance dispatch table
//! 6. Enumerates physical devices via vkEnumeratePhysicalDevices
//! 7. Reports GPU info via vkGetPhysicalDeviceProperties
//! 8. Cleans up via vkDestroyInstance
//!
//! If no Vulkan driver is installed, the test prints a notice and passes
//! (rather than failing CI on machines without Vulkan).

use spock::raw::bindings::*;
use spock::raw::{VkResultExt, VulkanLibrary};
use std::ffi::CStr;

#[test]
fn test_real_vulkan_loader() {
    // Step 1: Try to load the Vulkan library. If it's not installed, skip.
    let library = match VulkanLibrary::new() {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("SKIP: Vulkan library not available on this system: {}", e);
            return;
        }
    };
    println!("[OK] Loaded Vulkan shared library");

    // Step 2: Load entry-level dispatch table
    let entry = unsafe { library.load_entry() };
    println!("[OK] Loaded entry dispatch table");

    // Step 3: Query instance version via vkEnumerateInstanceVersion
    let enumerate_version = entry
        .vkEnumerateInstanceVersion
        .expect("vkEnumerateInstanceVersion should be available");

    let mut api_version: u32 = 0;
    unsafe { enumerate_version(&mut api_version) }
        .into_result()
        .expect("vkEnumerateInstanceVersion should succeed");

    let major = vk_api_version_major(api_version);
    let minor = vk_api_version_minor(api_version);
    let patch = vk_api_version_patch(api_version);
    println!("[OK] Driver supports Vulkan {}.{}.{}", major, minor, patch);
    assert!(major >= 1, "Vulkan major version should be at least 1");

    // Step 4: Create a real VkInstance
    let create_instance = entry
        .vkCreateInstance
        .expect("vkCreateInstance should be available");

    let app_name = c"spock-real-vulkan-test";
    let engine_name = c"spock";

    let app_info = VkApplicationInfo {
        sType: VkStructureType::STRUCTURE_TYPE_APPLICATION_INFO,
        pNext: std::ptr::null(),
        pApplicationName: app_name.as_ptr(),
        applicationVersion: vk_make_api_version(0, 0, 1, 0),
        pEngineName: engine_name.as_ptr(),
        engineVersion: vk_make_api_version(0, 0, 1, 0),
        apiVersion: VK_API_VERSION_1_0,
    };

    let create_info = VkInstanceCreateInfo {
        sType: VkStructureType::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        pApplicationInfo: &app_info,
        enabledLayerCount: 0,
        ppEnabledLayerNames: std::ptr::null(),
        enabledExtensionCount: 0,
        ppEnabledExtensionNames: std::ptr::null(),
    };

    let mut instance: VkInstance = std::ptr::null_mut();
    let result = unsafe { create_instance(&create_info, std::ptr::null(), &mut instance) };

    // ERROR_INCOMPATIBLE_DRIVER means the loader is present but no actual
    // Vulkan ICD (driver) is installed. This happens on the Windows GitHub
    // Actions runner, which ships vulkan-1.dll but no GPU driver. It's a
    // legitimate "no Vulkan available" outcome, not a test failure.
    if result == VkResult::ERROR_INCOMPATIBLE_DRIVER {
        eprintln!("SKIP: Vulkan loader is present but no compatible driver is installed");
        return;
    }

    assert_eq!(
        result,
        VkResult::SUCCESS,
        "vkCreateInstance should return SUCCESS, got {:?}",
        result
    );
    assert!(!instance.is_null(), "VkInstance should not be null");
    println!("[OK] Created VkInstance: {:p}", instance);

    // Step 5: Load instance dispatch table
    let instance_table = unsafe { library.load_instance(instance) };
    println!("[OK] Loaded instance dispatch table");

    // Step 6: Enumerate physical devices
    let enumerate_devices = instance_table
        .vkEnumeratePhysicalDevices
        .expect("vkEnumeratePhysicalDevices should be available");

    let mut device_count: u32 = 0;
    let result = unsafe { enumerate_devices(instance, &mut device_count, std::ptr::null_mut()) };
    assert_eq!(
        result,
        VkResult::SUCCESS,
        "vkEnumeratePhysicalDevices count query should succeed, got {:?}",
        result
    );
    println!("[OK] Found {} physical device(s)", device_count);

    if device_count == 0 {
        println!("NOTE: No physical devices available; skipping device-level checks");
    } else {
        let mut devices: Vec<VkPhysicalDevice> = vec![std::ptr::null_mut(); device_count as usize];
        let result =
            unsafe { enumerate_devices(instance, &mut device_count, devices.as_mut_ptr()) };
        assert_eq!(
            result,
            VkResult::SUCCESS,
            "vkEnumeratePhysicalDevices fetch should succeed, got {:?}",
            result
        );

        // Step 7: Get properties of each physical device
        let get_props = instance_table
            .vkGetPhysicalDeviceProperties
            .expect("vkGetPhysicalDeviceProperties should be available");

        for (i, &device) in devices.iter().enumerate() {
            assert!(
                !device.is_null(),
                "Physical device {} should not be null",
                i
            );
            let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
            unsafe { get_props(device, &mut props) };

            let device_name = unsafe {
                CStr::from_ptr(props.deviceName.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            let dev_major = vk_api_version_major(props.apiVersion);
            let dev_minor = vk_api_version_minor(props.apiVersion);
            let dev_patch = vk_api_version_patch(props.apiVersion);

            println!(
                "[OK] Device {}: \"{}\" (Vulkan {}.{}.{}, type={:?})",
                i, device_name, dev_major, dev_minor, dev_patch, props.deviceType
            );

            assert!(!device_name.is_empty(), "Device name should not be empty");
            assert!(dev_major >= 1, "Device API version major should be >= 1");
        }
    }

    // Step 8: Clean up
    let destroy_instance = instance_table
        .vkDestroyInstance
        .expect("vkDestroyInstance should be available");
    unsafe { destroy_instance(instance, std::ptr::null()) };
    println!("[OK] Destroyed VkInstance");

    println!("\n=== Real Vulkan loader test PASSED ===");
}
