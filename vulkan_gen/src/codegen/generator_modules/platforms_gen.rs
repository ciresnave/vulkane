// SPDX-License-Identifier: MIT OR Apache-2.0
//! Platforms generator module
//!
//! Generates Rust platform-specific code from platforms.json intermediate file

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorModule, GeneratorResult};
use crate::parser::vk_types::PlatformDefinition;

/// Generator module for Vulkan platforms
pub struct PlatformGenerator;

impl Default for PlatformGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Rust code for a single platform definition
    fn generate_platform(&self, platform: &PlatformDefinition) -> String {
        let mut code = String::new();

        // Documentation comment
        if let Some(comment) = &platform.comment {
            code.push_str(&format!("/// {}\n", comment));
        } else {
            code.push_str(&format!("/// Platform: {}\n", platform.name));
        }

        // Generate platform-specific code with conditional compilation
        let cfg_attr = self.map_platform_to_cfg(&platform.name);

        if !cfg_attr.is_empty() {
            code.push_str(&format!("#[cfg({})]\n", cfg_attr));
        }

        // Generate platform constant
        code.push_str(&format!(
            "pub const VK_PLATFORM_{}: &str = \"{}\";\n",
            platform.name.to_uppercase(),
            platform.name
        ));

        // Generate protection macro constant
        if !platform.protect.is_empty() {
            if !cfg_attr.is_empty() {
                code.push_str(&format!("#[cfg({})]\n", cfg_attr));
            }
            code.push_str(&format!(
                "pub const VK_PROTECT_{}: &str = \"{}\";\n",
                platform.name.to_uppercase(),
                platform.protect
            ));
        }

        // Add deprecation warning if applicable
        if let Some(deprecated) = &platform.deprecated {
            code.push_str(&format!("#[deprecated = \"{}\"]\n", deprecated));
        }

        code.push('\n');
        code
    }

    /// Map Vulkan platform names to Rust cfg attributes
    fn map_platform_to_cfg(&self, platform_name: &str) -> String {
        match platform_name.to_lowercase().as_str() {
            "win32" | "windows" => "target_os = \"windows\"".to_string(),
            "xlib" | "xcb" | "wayland" => "target_os = \"linux\"".to_string(),
            "android" => "target_os = \"android\"".to_string(),
            "macos" | "metal" => "target_os = \"macos\"".to_string(),
            "ios" => "target_os = \"ios\"".to_string(),
            "ggp" => "target_os = \"stadia\"".to_string(), // Google Game Platform
            _ => "".to_string(),                           // Platform-agnostic or unknown
        }
    }

    /// Generate platform detection functions
    fn generate_platform_helpers(&self) -> String {
        r#"
/// Check if the current platform supports the given Vulkan platform
pub fn is_platform_supported(platform: &str) -> bool {
    match platform.to_lowercase().as_str() {
        #[cfg(target_os = "windows")]
        "win32" | "windows" => true,

        #[cfg(target_os = "linux")]
        "xlib" | "xcb" | "wayland" => true,

        #[cfg(target_os = "android")]
        "android" => true,

        #[cfg(target_os = "macos")]
        "macos" | "metal" => true,

        #[cfg(target_os = "ios")]
        "ios" => true,

        _ => false,
    }
}

/// Get the primary platform identifier for the current target
pub fn get_current_platform() -> &'static str {
    #[cfg(target_os = "windows")]
    return "win32";

    #[cfg(target_os = "linux")]
    return "xlib";

    #[cfg(target_os = "android")]
    return "android";

    #[cfg(target_os = "macos")]
    return "macos";

    #[cfg(target_os = "ios")]
    return "ios";

    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    return "unknown";
}

"#
        .to_string()
    }
}

impl GeneratorModule for PlatformGenerator {
    fn name(&self) -> &str {
        "PlatformGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["platforms.json".to_string()]
    }

    fn output_file(&self) -> String {
        "platforms.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        vec![] // Platforms don't depend on other modules
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("platforms.json");
        let input_content = fs::read_to_string(input_path).map_err(GeneratorError::Io)?;

        // Parse JSON - direct array format
        let platforms: Vec<PlatformDefinition> =
            serde_json::from_str(&input_content).map_err(GeneratorError::Json)?;

        // Generate code
        let mut generated_code = String::new();

        // Don't add imports here - they're handled by the assembler

        // Add allow directives
        generated_code.push_str("#[allow(non_camel_case_types)]\n");
        generated_code.push_str("#[allow(dead_code)]\n\n");

        // Add platform helper functions first
        generated_code.push_str(&self.generate_platform_helpers());

        // Generate platform-specific code
        for platform in &platforms {
            generated_code.push_str(&self.generate_platform(platform));
        }

        // Write output file
        let output_path = output_dir.join(self.output_file());
        fs::write(output_path, generated_code).map_err(GeneratorError::Io)?;

        crate::codegen::logging::log_info(&format!(
            "PlatformGenerator: Generated {} platform definitions",
            platforms.len()
        ));
        Ok(())
    }
}
