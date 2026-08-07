//! Tags generator module
//!
//! Generates Rust tag definitions from tags.json intermediate file

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorModule, GeneratorResult};
use crate::parser::vk_types::TagDefinition;

/// Generator module for Vulkan vendor tags
pub struct TagGenerator;

impl Default for TagGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TagGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Rust code for a single tag definition
    fn generate_tag(&self, tag: &TagDefinition) -> String {
        let mut code = String::new();

        // Documentation comment with metadata
        if let Some(comment) = &tag.comment {
            code.push_str(&format!("/// {}\n", comment));
        } else {
            code.push_str(&format!("/// Vendor tag: {}\n", tag.name));
        }

        // Add author information
        code.push_str(&format!("/// Author: {}\n", tag.author));

        // Add contact if available
        if let Some(contact) = &tag.contact {
            code.push_str(&format!("/// Contact: {}\n", contact));
        }

        // Generate tag constant
        code.push_str(&format!(
            "pub const VK_TAG_{}: &str = \"{}\";\n",
            tag.name.to_uppercase(),
            tag.name
        ));

        // Generate author constant
        code.push_str(&format!(
            "pub const VK_TAG_{}_AUTHOR: &str = \"{}\";\n",
            tag.name.to_uppercase(),
            tag.author
        ));

        // Add contact constant if available
        if let Some(contact) = &tag.contact {
            code.push_str(&format!(
                "pub const VK_TAG_{}_CONTACT: &str = \"{}\";\n",
                tag.name.to_uppercase(),
                contact
            ));
        }

        // Add deprecation warning if applicable
        if let Some(deprecated) = &tag.deprecated {
            code.push_str(&format!("#[deprecated = \"{}\"]\n", deprecated));
        }

        code.push('\n');
        code
    }

    /// Generate tag registry and helper functions
    fn generate_tag_helpers(&self, tags: &[TagDefinition]) -> String {
        let mut code = String::new();

        // Generate tag registry structure
        code.push_str(
            r#"
/// Vulkan vendor tag information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanTag {
    pub name: &'static str,
    pub author: &'static str,
    pub contact: Option<&'static str>,
}

impl VulkanTag {
    // Note: there is deliberately no `is_deprecated` here. `vk.xml` `<tag>`
    // elements carry only `name`, `author` and `contact` — deprecation is a
    // property of *extensions* (`deprecatedby`), never of vendor tags, so the
    // question has no answer to return rather than an unimplemented one.

    /// Get the full tag identifier
    pub fn full_name(&self) -> String {
        format!("VK_{}", self.name)
    }
}

"#,
        );

        // Generate tag registry array
        code.push_str("/// Registry of all known Vulkan vendor tags\n");
        code.push_str("pub static VULKAN_TAGS: &[VulkanTag] = &[\n");

        for tag in tags {
            let contact_str = if let Some(contact) = &tag.contact {
                format!("Some(\"{}\")", contact)
            } else {
                "None".to_string()
            };

            code.push_str(&format!(
                "    VulkanTag {{ name: \"{}\", author: \"{}\", contact: {} }},\n",
                tag.name, tag.author, contact_str
            ));
        }

        code.push_str("];\n\n");

        // Generate helper functions
        code.push_str(
            r#"
/// Find a tag by name
pub fn find_tag(name: &str) -> Option<&'static VulkanTag> {
    VULKAN_TAGS.iter().find(|tag| tag.name == name)
}

/// Check if a tag exists
pub fn is_valid_tag(name: &str) -> bool {
    find_tag(name).is_some()
}

/// Get all tags by a specific author
pub fn get_tags_by_author(author: &str) -> Vec<&'static VulkanTag> {
    VULKAN_TAGS.iter().filter(|tag| tag.author == author).collect()
}

/// Get all available tag names
pub fn get_all_tag_names() -> Vec<&'static str> {
    VULKAN_TAGS.iter().map(|tag| tag.name).collect()
}

"#,
        );

        code
    }
}

impl GeneratorModule for TagGenerator {
    fn name(&self) -> &str {
        "TagGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["tags.json".to_string()]
    }

    fn output_file(&self) -> String {
        "tags.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        vec![] // Tags don't depend on other modules
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("tags.json");
        let input_content = fs::read_to_string(input_path).map_err(GeneratorError::Io)?;

        // Parse JSON - direct array format
        let tags: Vec<TagDefinition> =
            serde_json::from_str(&input_content).map_err(GeneratorError::Json)?;

        // Generate code
        let mut generated_code = String::new();

        // Don't add imports here - they're handled by the assembler

        // Add allow directives
        generated_code.push_str("#[allow(non_camel_case_types)]\n");
        generated_code.push_str("#[allow(dead_code)]\n\n");

        // Generate tag helper functions and registry
        generated_code.push_str(&self.generate_tag_helpers(&tags));

        // Generate individual tag constants
        for tag in &tags {
            generated_code.push_str(&self.generate_tag(tag));
        }

        // Write output file
        let output_path = output_dir.join(self.output_file());
        fs::write(output_path, generated_code).map_err(GeneratorError::Io)?;

        crate::codegen::logging::log_info(&format!(
            "TagGenerator: Generated {} tag definitions",
            tags.len()
        ));
        Ok(())
    }
}
