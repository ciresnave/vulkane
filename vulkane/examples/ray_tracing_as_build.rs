//! End-to-end `VK_KHR_acceleration_structure` build using the vulkane
//! safe wrappers.
//!
//! Builds a complete two-level acceleration structure from scratch —
//! one AABB-geometry BLAS, one TLAS instancing it — and prints the
//! device addresses that a ray-query or ray-tracing-pipeline shader
//! would bind to. No SPIR-V, no trace: this example is the proof the
//! *build* surface is ergonomic and correct. Shader-side traversal is
//! orthogonal (enable `rayQuery` and `accelerationStructure` features;
//! call `rayQueryEXT` against the TLAS device address in your compute
//! shader).
//!
//! The workflow:
//!
//! 1. Enable the five required extensions (`buffer_device_address`,
//!    `acceleration_structure`, `deferred_host_operations`, plus the
//!    two instance-level prereqs) and corresponding feature bits.
//! 2. Allocate an AABB primitive buffer + scratch buffer, each with
//!    `SHADER_DEVICE_ADDRESS` usage so they have GPU virtual addresses.
//! 3. Ask the driver how big the BLAS and its build scratch need to be
//!    via [`Device::acceleration_structure_build_sizes`].
//! 4. Allocate the BLAS backing buffer at the reported size, create
//!    the AS handle via [`AccelerationStructure::new`], then record
//!    the build via [`CommandBufferRecording::build_acceleration_structure`].
//! 5. Repeat for the TLAS: one `VkAccelerationStructureInstanceKHR`
//!    pointing at the BLAS device address, one scratch allocation,
//!    one build command.
//! 6. Submit, wait, and print both device addresses.
//!
//! Skips cleanly if the device doesn't expose
//! `VK_KHR_acceleration_structure`.
//!
//! Run with: `cargo run --example ray_tracing_as_build -p vulkane`

use std::sync::Arc;
use vulkane::raw::bindings::{VkAabbPositionsKHR, VkAccelerationStructureInstanceKHR};
use vulkane::safe::{
    AccelerationStructure, AccelerationStructureBuildFlags, AccelerationStructureBuildMode,
    AccelerationStructureBuildType, AccelerationStructureCreateInfo, AccelerationStructureGeometry,
    AccelerationStructureType, ApiVersion, Buffer, BufferCreateInfo, BufferUsage, BuildRange,
    DeviceCreateInfo, DeviceExtensions, DeviceFeatures, DeviceMemory, Instance, InstanceCreateInfo,
    InstanceExtensions, MemoryAllocateInfo, MemoryPropertyFlags, QueueCreateInfo, QueueFlags,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Instance with the extension prerequisites.
    let instance_exts = InstanceExtensions::new().khr_get_physical_device_properties2();
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane ray-tracing AS build"),
        api_version: ApiVersion::V1_2,
        enabled_extensions: Some(&instance_exts),
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: cannot create Vulkan instance: {e:?}");
            return Ok(());
        }
    };

    // Pick a compute-capable GPU.
    let Some(physical) = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())
    else {
        eprintln!("SKIP: no compute-capable GPU");
        return Ok(());
    };
    let qf = physical.find_queue_family(QueueFlags::COMPUTE).unwrap();
    println!("Using GPU: {}", physical.properties().device_name());

    // Probe which of the ray-tracing extensions this device actually
    // exposes before we ask for them — create_device will fail hard on
    // any missing entry.
    let available: std::collections::HashSet<String> = physical
        .enumerate_extension_properties()
        .unwrap_or_default()
        .into_iter()
        .map(|p| p.name().to_string())
        .collect();
    let required = [
        "VK_KHR_acceleration_structure",
        "VK_KHR_deferred_host_operations",
    ];
    for name in &required {
        if !available.contains(*name) {
            eprintln!("SKIP: device does not expose {name}");
            return Ok(());
        }
    }

    // Device: the ray-tracing extension cluster. `buffer_device_address`
    // is core in Vulkan 1.2 so we don't list the KHR extension name;
    // the feature bit is what matters.
    let dev_exts = DeviceExtensions::new()
        .khr_acceleration_structure()
        .khr_deferred_host_operations();
    let features = DeviceFeatures::new()
        .with_buffer_device_address()
        .with_acceleration_structure()
        .with_ray_query();
    let device = match physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(qf)],
        enabled_extensions: Some(&dev_exts),
        enabled_features: Some(&features),
        ..Default::default()
    }) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: acceleration-structure extension not supported on this device: {e:?}");
            return Ok(());
        }
    };

    // Helper: allocate a host-visible buffer with SHADER_DEVICE_ADDRESS
    // + the given usage, write `bytes` into it, and return
    // (Buffer, DeviceMemory, device_address). The allocation uses the
    // `VkMemoryAllocateFlagsInfo` pNext chain with
    // `DEVICE_ADDRESS_BIT` so the returned address is valid.
    fn alloc_host_visible_with_addr(
        device: &vulkane::safe::Device,
        physical: &vulkane::safe::PhysicalDevice,
        bytes: &[u8],
        usage: BufferUsage,
    ) -> Result<(Buffer, DeviceMemory, u64), Box<dyn std::error::Error>> {
        use vulkane::raw::PNextChainable;
        use vulkane::raw::bindings::{
            MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, VkMemoryAllocateFlagsInfo,
        };
        use vulkane::safe::PNextChain;

        let buf_size = bytes.len().max(1) as u64;
        let buffer = Buffer::new(
            device,
            BufferCreateInfo {
                size: buf_size,
                usage: usage | BufferUsage::SHADER_DEVICE_ADDRESS,
            },
        )?;
        let req = buffer.memory_requirements();
        let mt = physical
            .find_memory_type(
                req.memory_type_bits,
                MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or("no host-visible memory type")?;
        let mut chain = PNextChain::new();
        let mut flags_info = VkMemoryAllocateFlagsInfo::new_pnext();
        flags_info.flags = MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        chain.push(flags_info);
        let mut memory = DeviceMemory::allocate_with(
            device,
            &MemoryAllocateInfo {
                size: req.size,
                memory_type_index: mt,
                pnext: Some(&chain),
                priority: None,
            },
        )?;
        buffer.bind_memory(&memory, 0)?;
        if !bytes.is_empty() {
            let mut mapped = memory.map()?;
            mapped.as_slice_mut()[..bytes.len()].copy_from_slice(bytes);
        }
        let addr = buffer.device_address()?;
        Ok((buffer, memory, addr))
    }

    // ── BLAS: one AABB primitive ─────────────────────────────────────
    let aabb = VkAabbPositionsKHR {
        minX: -1.0,
        minY: -1.0,
        minZ: -1.0,
        maxX: 1.0,
        maxY: 1.0,
        maxZ: 1.0,
    };
    let aabb_bytes = unsafe {
        std::slice::from_raw_parts(
            &aabb as *const _ as *const u8,
            std::mem::size_of::<VkAabbPositionsKHR>(),
        )
    };
    let (_aabb_buf, _aabb_mem, aabb_addr) = alloc_host_visible_with_addr(
        &device,
        &physical,
        aabb_bytes,
        BufferUsage(0x0008_0000), // ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
    )?;

    let blas_geom = AccelerationStructureGeometry::Aabbs {
        data_address: aabb_addr,
        stride: std::mem::size_of::<VkAabbPositionsKHR>() as u64,
    };
    let blas_sizes = device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        AccelerationStructureType::BottomLevel,
        &[blas_geom],
        &[1],
        AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
    )?;
    println!(
        "BLAS: structure_size={} build_scratch={}",
        blas_sizes.acceleration_structure_size, blas_sizes.build_scratch_size
    );

    // BLAS backing buffer — ACCELERATION_STRUCTURE_STORAGE usage.
    let blas_backing = Arc::new(Buffer::new(
        &device,
        BufferCreateInfo {
            size: blas_sizes.acceleration_structure_size,
            usage: BufferUsage(0x0010_0000) // ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                | BufferUsage::SHADER_DEVICE_ADDRESS,
        },
    )?);
    let req = blas_backing.memory_requirements();
    let mt = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no device-local memory type for BLAS")?;
    let blas_backing_mem = DeviceMemory::allocate(&device, req.size, mt)?;
    blas_backing.bind_memory(&blas_backing_mem, 0)?;

    let blas = AccelerationStructure::new(
        &device,
        AccelerationStructureCreateInfo {
            buffer: Arc::clone(&blas_backing),
            offset: 0,
            size: blas_sizes.acceleration_structure_size,
            type_: AccelerationStructureType::BottomLevel,
            _marker: std::marker::PhantomData,
        },
    )?;

    // Scratch for the BLAS build.
    let blas_scratch = Buffer::new(
        &device,
        BufferCreateInfo {
            size: blas_sizes.build_scratch_size.max(1),
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
        },
    )?;
    let req = blas_scratch.memory_requirements();
    let mt = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no device-local memory type for BLAS scratch")?;
    let blas_scratch_mem = DeviceMemory::allocate(&device, req.size, mt)?;
    blas_scratch.bind_memory(&blas_scratch_mem, 0)?;
    let blas_scratch_addr = blas_scratch.device_address()?;

    // Record the BLAS build.
    let queue = device.get_queue(qf, 0);
    // Build the BLAS first; host-side waitIdle between one_shot calls
    // is a simple synchronisation point between BLAS and TLAS builds.
    // A production-quality path would chain the two builds in one
    // command buffer with a sync2 AS→AS memory barrier between them.
    queue.one_shot(&device, qf, |rec| {
        rec.build_acceleration_structure(
            AccelerationStructureType::BottomLevel,
            AccelerationStructureBuildMode::Build,
            AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
            &blas,
            None,
            &[blas_geom],
            &[BuildRange {
                primitive_count: 1,
                ..Default::default()
            }],
            blas_scratch_addr,
        )
    })?;

    let blas_addr = blas.device_address()?;
    println!("BLAS device address: 0x{:016x}", blas_addr);

    // ── TLAS: one instance pointing at the BLAS ─────────────────────
    let instance = VkAccelerationStructureInstanceKHR {
        // Identity transform.
        transform: unsafe { std::mem::zeroed() },
        instanceCustomIndex: 0,
        mask: 0xFF,
        instanceShaderBindingTableRecordOffset: 0,
        flags: 0,
        accelerationStructureReference: blas_addr,
    };
    // Apply the identity transform by hand: 3×4 row-major with
    // diagonal=1 (VkTransformMatrixKHR.matrix is [[f32; 4]; 3]).
    let mut instance = instance;
    instance.transform.matrix[0][0] = 1.0;
    instance.transform.matrix[1][1] = 1.0;
    instance.transform.matrix[2][2] = 1.0;

    let inst_bytes = unsafe {
        std::slice::from_raw_parts(
            &instance as *const _ as *const u8,
            std::mem::size_of::<VkAccelerationStructureInstanceKHR>(),
        )
    };
    let (_inst_buf, _inst_mem, inst_addr) = alloc_host_visible_with_addr(
        &device,
        &physical,
        inst_bytes,
        BufferUsage(0x0008_0000), // ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
    )?;

    let tlas_geom = AccelerationStructureGeometry::Instances {
        data_address: inst_addr,
        array_of_pointers: false,
    };
    let tlas_sizes = device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        AccelerationStructureType::TopLevel,
        &[tlas_geom],
        &[1],
        AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
    )?;
    println!(
        "TLAS: structure_size={} build_scratch={}",
        tlas_sizes.acceleration_structure_size, tlas_sizes.build_scratch_size
    );

    let tlas_backing = Arc::new(Buffer::new(
        &device,
        BufferCreateInfo {
            size: tlas_sizes.acceleration_structure_size,
            usage: BufferUsage(0x0010_0000) | BufferUsage::SHADER_DEVICE_ADDRESS,
        },
    )?);
    let req = tlas_backing.memory_requirements();
    let mt = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no device-local memory type for TLAS")?;
    let tlas_backing_mem = DeviceMemory::allocate(&device, req.size, mt)?;
    tlas_backing.bind_memory(&tlas_backing_mem, 0)?;

    let tlas = AccelerationStructure::new(
        &device,
        AccelerationStructureCreateInfo {
            buffer: Arc::clone(&tlas_backing),
            offset: 0,
            size: tlas_sizes.acceleration_structure_size,
            type_: AccelerationStructureType::TopLevel,
            _marker: std::marker::PhantomData,
        },
    )?;

    let tlas_scratch = Buffer::new(
        &device,
        BufferCreateInfo {
            size: tlas_sizes.build_scratch_size.max(1),
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
        },
    )?;
    let req = tlas_scratch.memory_requirements();
    let mt = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no device-local memory type for TLAS scratch")?;
    let tlas_scratch_mem = DeviceMemory::allocate(&device, req.size, mt)?;
    tlas_scratch.bind_memory(&tlas_scratch_mem, 0)?;
    let tlas_scratch_addr = tlas_scratch.device_address()?;

    queue.one_shot(&device, qf, |rec| {
        rec.build_acceleration_structure(
            AccelerationStructureType::TopLevel,
            AccelerationStructureBuildMode::Build,
            AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
            &tlas,
            None,
            &[tlas_geom],
            &[BuildRange {
                primitive_count: 1,
                ..Default::default()
            }],
            tlas_scratch_addr,
        )
    })?;

    let tlas_addr = tlas.device_address()?;
    println!("TLAS device address: 0x{:016x}", tlas_addr);
    println!(
        "\nTo use from a compute shader: bind the TLAS address into an \
         `accelerationStructureEXT` descriptor or a `rayQueryEXT` initializer \
         and issue `rayQueryInitializeEXT` / `rayQueryProceedEXT` / \
         `rayQueryGetIntersectionTypeEXT`. Enable the `rayQuery` feature bit \
         (already on here) and compile the shader with GL_EXT_ray_query."
    );

    Ok(())
}
