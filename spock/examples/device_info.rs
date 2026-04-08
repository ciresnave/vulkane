//! Spock example: enumerate Vulkan devices and print detailed information.
//!
//! Run with: `cargo run --example device_info -p spock --features fetch-spec`
//!
//! This example demonstrates the full spock loader flow:
//! 1. Loading the Vulkan library
//! 2. Loading entry-level, instance-level, and device-level dispatch tables
//! 3. Querying physical device properties, memory, and queue families
//! 4. Creating a logical device with a graphics queue
//! 5. Cleaning up

use spock::raw::bindings::*;
use spock::raw::{VkResultExt, VulkanLibrary};
use std::ffi::CStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the Vulkan shared library
    let library = VulkanLibrary::new().map_err(|e| format!("Failed to load Vulkan: {}", e))?;
    println!("Loaded Vulkan library\n");

    // Load entry-level functions (no instance needed)
    let entry = unsafe { library.load_entry() };

    // Print the driver's supported instance API version
    if let Some(ev) = entry.vkEnumerateInstanceVersion {
        let mut version = 0u32;
        unsafe { ev(&mut version) }.into_result()?;
        println!(
            "Driver supports Vulkan {}.{}.{}",
            vk_api_version_major(version),
            vk_api_version_minor(version),
            vk_api_version_patch(version)
        );
    }

    // Create an instance
    let instance = unsafe { create_instance(&entry)? };
    println!("Created VkInstance\n");

    // Load instance-level dispatch table for the new instance
    let inst_table = unsafe { library.load_instance(instance) };

    // Enumerate physical devices
    let physical_devices = unsafe { enumerate_physical_devices(&inst_table, instance)? };
    println!("Found {} physical device(s)\n", physical_devices.len());

    for (idx, &phys) in physical_devices.iter().enumerate() {
        println!("=== Physical device {} ===", idx);
        unsafe {
            print_device_info(&inst_table, phys);
        }
        println!();

        // Try to create a logical device with the first queue family
        if let Some(queue_family) = unsafe { find_first_queue_family(&inst_table, phys) } {
            println!(
                "Creating logical device using queue family {} on physical device {}",
                queue_family, idx
            );
            match unsafe { create_logical_device(&inst_table, phys, queue_family) } {
                Ok(device) => {
                    println!("Created VkDevice");

                    // Load device-level dispatch table
                    let dev_table = unsafe { library.load_device(instance, device) };
                    println!(
                        "Loaded device dispatch table (vkDestroyDevice loaded: {})",
                        dev_table.vkDestroyDevice.is_some()
                    );

                    // Clean up the device
                    if let Some(destroy) = dev_table.vkDestroyDevice {
                        unsafe { destroy(device, std::ptr::null()) };
                        println!("Destroyed VkDevice");
                    }
                }
                Err(e) => {
                    println!("Failed to create logical device: {:?}", e);
                }
            }
            println!();
        }
    }

    // Clean up the instance
    if let Some(destroy) = inst_table.vkDestroyInstance {
        unsafe { destroy(instance, std::ptr::null()) };
        println!("Destroyed VkInstance");
    }

    Ok(())
}

unsafe fn create_instance(
    entry: &VkEntryDispatchTable,
) -> Result<VkInstance, Box<dyn std::error::Error>> {
    let create = entry
        .vkCreateInstance
        .ok_or("vkCreateInstance not available")?;

    let app_name = c"spock device_info example";
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
    unsafe { create(&create_info, std::ptr::null(), &mut instance) }.into_result()?;
    Ok(instance)
}

unsafe fn enumerate_physical_devices(
    inst: &VkInstanceDispatchTable,
    instance: VkInstance,
) -> Result<Vec<VkPhysicalDevice>, Box<dyn std::error::Error>> {
    let enumerate = inst
        .vkEnumeratePhysicalDevices
        .ok_or("vkEnumeratePhysicalDevices not available")?;

    let mut count = 0u32;
    unsafe { enumerate(instance, &mut count, std::ptr::null_mut()) }.into_result()?;
    let mut devices = vec![std::ptr::null_mut(); count as usize];
    unsafe { enumerate(instance, &mut count, devices.as_mut_ptr()) }.into_result()?;
    Ok(devices)
}

unsafe fn print_device_info(inst: &VkInstanceDispatchTable, phys: VkPhysicalDevice) {
    if let Some(get_props) = inst.vkGetPhysicalDeviceProperties {
        let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
        unsafe { get_props(phys, &mut props) };

        let name = unsafe { CStr::from_ptr(props.deviceName.as_ptr()).to_string_lossy() };
        println!("  Name:           {}", name);
        println!(
            "  API version:    {}.{}.{}",
            vk_api_version_major(props.apiVersion),
            vk_api_version_minor(props.apiVersion),
            vk_api_version_patch(props.apiVersion)
        );
        println!("  Driver version: 0x{:08x}", props.driverVersion);
        println!("  Vendor ID:      0x{:04x}", props.vendorID);
        println!("  Device type:    {:?}", props.deviceType);
    }

    // Memory properties
    if let Some(get_mem) = inst.vkGetPhysicalDeviceMemoryProperties {
        let mut mem: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
        unsafe { get_mem(phys, &mut mem) };
        println!(
            "  Memory:         {} heap(s), {} type(s)",
            mem.memoryHeapCount, mem.memoryTypeCount
        );
        for i in 0..mem.memoryHeapCount as usize {
            let heap = &mem.memoryHeaps[i];
            let mb = heap.size / (1024 * 1024);
            println!("    Heap {}: {} MB (flags=0x{:x})", i, mb, heap.flags);
        }
    }

    // Queue families
    if let Some(get_qf) = inst.vkGetPhysicalDeviceQueueFamilyProperties {
        let mut count = 0u32;
        unsafe { get_qf(phys, &mut count, std::ptr::null_mut()) };
        let mut families: Vec<VkQueueFamilyProperties> =
            vec![unsafe { std::mem::zeroed() }; count as usize];
        unsafe { get_qf(phys, &mut count, families.as_mut_ptr()) };
        println!("  Queue families: {}", count);
        for (i, qf) in families.iter().enumerate() {
            println!(
                "    Family {}: {} queue(s), flags=0x{:x}",
                i, qf.queueCount, qf.queueFlags
            );
        }
    }
}

unsafe fn find_first_queue_family(
    inst: &VkInstanceDispatchTable,
    phys: VkPhysicalDevice,
) -> Option<u32> {
    let get_qf = inst.vkGetPhysicalDeviceQueueFamilyProperties?;
    let mut count = 0u32;
    unsafe { get_qf(phys, &mut count, std::ptr::null_mut()) };
    if count > 0 { Some(0) } else { None }
}

unsafe fn create_logical_device(
    inst: &VkInstanceDispatchTable,
    phys: VkPhysicalDevice,
    queue_family_index: u32,
) -> Result<VkDevice, VkResult> {
    let create = inst.vkCreateDevice.expect("vkCreateDevice not available");

    let queue_priority: f32 = 1.0;
    let queue_create = VkDeviceQueueCreateInfo {
        sType: VkStructureType::STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        queueFamilyIndex: queue_family_index,
        queueCount: 1,
        pQueuePriorities: &queue_priority,
    };

    let device_create = VkDeviceCreateInfo {
        sType: VkStructureType::STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        queueCreateInfoCount: 1,
        pQueueCreateInfos: &queue_create,
        enabledLayerCount: 0,
        ppEnabledLayerNames: std::ptr::null(),
        enabledExtensionCount: 0,
        ppEnabledExtensionNames: std::ptr::null(),
        pEnabledFeatures: std::ptr::null(),
    };

    let mut device: VkDevice = std::ptr::null_mut();
    unsafe { create(phys, &device_create, std::ptr::null(), &mut device) }.into_result()?;
    Ok(device)
}
