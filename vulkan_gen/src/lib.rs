pub mod codegen;
pub mod parser;
pub mod vulkan_spec_parser;

pub use vulkan_spec_parser::{VulkanSpecification, parse_vulkan_spec};

// Re-export the public codegen API
pub use codegen::assembler::{AssemblerResult, CodeAssembler};
pub use codegen::generator_modules;

/// Generate complete Vulkan bindings from XML specification to a Rust source file.
///
/// This is the main entry point. It parses vk.xml, generates intermediate JSON,
/// then generates Rust code from the JSON.
pub fn generate_bindings(
    xml_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let temp_dir = tempfile::TempDir::new()?;
    let intermediate_dir = temp_dir.path();

    // Step 1: Parse XML to intermediate JSON
    parse_vulkan_spec(xml_path, intermediate_dir)?;

    // Step 2: Generate Rust from JSON
    generate_rust_bindings(intermediate_dir, output_path)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;

    // Step 3: Generate the safe-layer companion files (Layer 1 / 3).
    // These live alongside `vulkan_bindings.rs` and are `include!`'d
    // from the vulkane `safe` module.
    let out_dir = output_path
        .parent()
        .ok_or("output_path must have a parent directory")?;

    let features_out = out_dir.join("device_features_generated.rs");
    let n_features =
        codegen::generator_modules::device_features_gen::generate_device_features(
            intermediate_dir,
            &features_out,
        )
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
    codegen::logging::log_info(&format!(
        "Generated DeviceFeatures with {} feature-bit methods",
        n_features
    ));

    let ext_stats =
        codegen::generator_modules::extensions_builder_gen::generate_extensions_builders(
            intermediate_dir,
            out_dir,
        )
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
    codegen::logging::log_info(&format!(
        "Generated DeviceExtensions ({} methods) and InstanceExtensions ({} methods)",
        ext_stats.device_methods, ext_stats.instance_methods
    ));

    let handles_out = out_dir.join("auto_handles_generated.rs");
    let handle_stats = codegen::generator_modules::safe_handles_gen::generate_safe_handles(
        intermediate_dir,
        &handles_out,
    )
    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
    codegen::logging::log_info(&format!(
        "Generated {} auto-RAII handle wrappers ({} skipped)",
        handle_stats.generated, handle_stats.skipped
    ));

    let cmd_stats = codegen::generator_modules::safe_commands_gen::generate_safe_commands(
        intermediate_dir,
        out_dir,
    )
    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
    codegen::logging::log_info(&format!(
        "Generated ext-trait methods: Device={}, Instance={}, PhysicalDevice={}, Queue={}, CommandBuffer={} ({} skipped)",
        cmd_stats.device_methods,
        cmd_stats.instance_methods,
        cmd_stats.physical_device_methods,
        cmd_stats.queue_methods,
        cmd_stats.command_buffer_methods,
        cmd_stats.skipped
    ));

    Ok(())
}

/// Generate Rust bindings from intermediate JSON files (step 2 only).
pub fn generate_rust_bindings(
    intermediate_dir: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<(), codegen::CodegenError> {
    use codegen::generator_modules::*;

    codegen::logging::log_info("Starting Vulkan code generation");

    if !intermediate_dir.exists() {
        return Err(codegen::CodegenError::InvalidInput {
            message: format!(
                "Intermediate directory does not exist: {}",
                intermediate_dir.display()
            ),
        });
    }

    // Check data consistency
    if let Err(e) = codegen::type_integration::check_data_consistency(intermediate_dir) {
        return Err(codegen::CodegenError::InvalidInput {
            message: format!("Data consistency check failed: {}", e),
        });
    }

    // Create assembler and register all generator modules
    let mut assembler = CodeAssembler::new();
    assembler.register_module(Box::new(IncludeGenerator::new()));
    assembler.register_module(Box::new(MacroGenerator::new()));
    assembler.register_module(Box::new(ConstantGenerator::new()));
    assembler.register_module(Box::new(TypeGenerator::new()));
    assembler.register_module(Box::new(EnumGenerator::new()));
    assembler.register_module(Box::new(StructGenerator::new()));
    assembler.register_module(Box::new(FunctionGenerator::new()));
    assembler.register_module(Box::new(ExtensionGenerator::new()));

    // Assemble final file
    let output_dir = output_path.parent().unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(output_dir)?;

    // Generate fragments and assemble
    assembler.generate_fragments(intermediate_dir, output_dir)?;
    assembler.assemble_final_bindings(output_path)?;

    codegen::logging::log_info(&format!("Generated bindings to {}", output_path.display()));
    Ok(())
}
