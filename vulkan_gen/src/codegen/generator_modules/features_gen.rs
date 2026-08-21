//! Features generator module
//!
//! Generates Rust feature definitions from features.json intermediate file

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorModule, GeneratorResult};
use crate::parser::vk_types::FeatureDefinition;

/// Generator module for Vulkan features (API versions)
pub struct FeatureGenerator;

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Rust code for a single feature definition
    fn generate_feature(&self, feature: &FeatureDefinition) -> String {
        let mut code = String::new();

        // Documentation comment
        if let Some(comment) = &feature.comment {
            code.push_str(&format!("/// {}\n", comment));
        } else {
            code.push_str(&format!("/// Vulkan API feature: {}\n", feature.name));
        }

        // Generate API version constant
        if feature.name.starts_with("VK_VERSION_") {
            let version_parts = self.parse_version_number(&feature.number);
            if let Some((major, minor)) = version_parts {
                code.push_str(&format!(
                    "pub const {}: u32 = vk_make_api_version(0, {}, {}, 0);\n",
                    feature.name, major, minor
                ));
            } else {
                // Fallback for non-standard version formats
                code.push_str(&format!(
                    "pub const {}: u32 = {};\n",
                    feature.name, feature.number
                ));
            }
        } else {
            // Non-version features as regular constants
            code.push_str(&format!(
                "pub const {}: &str = \"{}\";\n",
                feature.name, feature.number
            ));
        }

        // Add deprecation warning if applicable
        if let Some(deprecated) = &feature.deprecated {
            code.push_str(&format!("#[deprecated = \"{}\"]\n", deprecated));
        }

        code.push('\n');
        code
    }

    /// Parse version number string like "1.2" into (major, minor)
    fn parse_version_number(&self, version: &str) -> Option<(u32, u32)> {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() >= 2
            && let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
        {
            return Some((major, minor));
        }
        None
    }

    /// Generate helper functions for version checking
    fn generate_version_helpers(&self) -> String {
        r#"
/// Helper function to create API version from components
pub const fn vk_make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

/// Extract major version from API version
pub const fn vk_api_version_major(version: u32) -> u32 {
    (version >> 22) & 0x7F
}

/// Extract minor version from API version
pub const fn vk_api_version_minor(version: u32) -> u32 {
    (version >> 12) & 0x3FF
}

/// Extract patch version from API version
pub const fn vk_api_version_patch(version: u32) -> u32 {
    version & 0xFFF
}

/// Extract variant from API version
pub const fn vk_api_version_variant(version: u32) -> u32 {
    version >> 29
}

"#
        .to_string()
    }
}

impl GeneratorModule for FeatureGenerator {
    fn name(&self) -> &str {
        "FeatureGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["features.json".to_string()]
    }

    fn output_file(&self) -> String {
        "features.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        vec![] // Features don't depend on other modules
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("features.json");
        let input_content = fs::read_to_string(input_path).map_err(GeneratorError::Io)?;

        // Parse JSON - direct array format
        let features: Vec<FeatureDefinition> =
            serde_json::from_str(&input_content).map_err(GeneratorError::Json)?;

        // Generate code
        let mut generated_code = String::new();

        // Don't add imports here - they're handled by the assembler

        // Add allow directives
        generated_code.push_str("#[allow(non_camel_case_types)]\n");
        generated_code.push_str("#[allow(dead_code)]\n\n");

        // Add version helper functions first
        generated_code.push_str(&self.generate_version_helpers());

        // Generate feature constants
        for feature in &features {
            generated_code.push_str(&self.generate_feature(feature));
        }

        // Write output file
        let output_path = output_dir.join(self.output_file());
        fs::write(output_path, generated_code).map_err(GeneratorError::Io)?;

        crate::codegen::logging::log_info(&format!(
            "FeatureGenerator: Generated {} feature definitions",
            features.len()
        ));
        Ok(())
    }
}
