// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vulkan Specification Parser
//!
//! Parses vk.xml and writes intermediate JSON files for the code generator.

use crate::parser::vk_types::*;
use std::fs;
use std::path::Path;

/// Complete Vulkan specification data
#[derive(Debug, Default)]
pub struct VulkanSpecification {
    pub constants: Vec<VulkanConstant>,
    pub enums: Vec<VulkanEnum>,
    pub structs: Vec<VulkanStruct>,
    pub types: Vec<VulkanType>,
    pub extensions: Vec<VulkanExtension>,
    pub functions: Vec<VulkanCommand>,
    pub features: Vec<VulkanFeature>,
    pub includes: Vec<VulkanInclude>,
    pub macros: Vec<VulkanMacro>,
    pub platforms: Vec<VulkanPlatform>,
    pub tags: Vec<VulkanTag>,
}

/// Parse Vulkan XML specification and write intermediate JSON files to output directory.
/// Uses the tree-based parser (roxmltree) for reliable nested element extraction.
pub fn parse_vulkan_spec<P: AsRef<Path>>(
    xml_path: P,
    output_dir: P,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let xml_path = xml_path.as_ref();
    let output_dir = output_dir.as_ref();

    if !xml_path.exists() {
        return Err(format!("XML file does not exist: {}", xml_path.display()).into());
    }

    fs::create_dir_all(output_dir)?;

    // Read and parse with tree parser
    let xml_content = fs::read_to_string(xml_path)?;
    let spec = crate::parser::tree_parser::parse_vk_xml(&xml_content)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    // Write each category to its JSON file
    macro_rules! write_json {
        ($filename:expr, $data:expr) => {
            fs::write(
                output_dir.join($filename),
                serde_json::to_string_pretty(&$data)?,
            )?;
        };
    }

    write_json!("constants.json", spec.constants);
    write_json!("enums.json", spec.enums);
    write_json!("structs.json", spec.structs);
    write_json!("types.json", spec.types);
    write_json!("functions.json", spec.functions);
    write_json!("extensions.json", spec.extensions);
    write_json!("features.json", spec.features);
    write_json!("includes.json", spec.includes);
    write_json!("macros.json", spec.macros);
    write_json!("platforms.json", spec.platforms);
    write_json!("tags.json", spec.tags);

    println!(
        "Successfully parsed Vulkan XML specification (tree parser) -> {}",
        output_dir.display()
    );

    Ok(())
}
