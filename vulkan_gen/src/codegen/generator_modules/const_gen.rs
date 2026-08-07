//! Constants generator module
//!
//! Generates Rust constants from constants.json intermediate file

use crate::codegen::logging::log_info;
use crate::parser::vk_types::ConstantDefinition;
use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorMetadata, GeneratorModule, GeneratorResult};

/// Generator module for Vulkan constants
pub struct ConstantGenerator;

impl Default for ConstantGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Rust code for a single constant
    fn generate_constant(&self, constant: &ConstantDefinition) -> String {
        // Handle enhanced constant definition with optional value
        let value = constant.value.as_deref().unwrap_or("0");
        let rust_type = self.infer_type_from_value(value);
        let rust_value = self.map_value(value, &rust_type);

        // Include documentation if available
        let mut doc_comment = format!("/// Vulkan constant: {}", constant.name);
        if let Some(ref comment) = constant.comment {
            doc_comment.push_str(&format!("\n/// {}", comment));
        }

        format!(
            "{}\npub const {}: {} = {};\n",
            doc_comment, constant.name, rust_type, rust_value
        )
    }

    /// Infer the type from the value for simplified constants
    fn infer_type_from_value(&self, value: &str) -> String {
        // Basic type inference from value format
        // Trim surrounding whitespace and parentheses
        let v = value.trim();
        // Strip a single pair of surrounding parentheses repeatedly
        let mut v_unparen = v;
        while v_unparen.starts_with('(') && v_unparen.ends_with(')') {
            v_unparen = v_unparen.trim_start_matches('(').trim_end_matches(')');
            v_unparen = v_unparen.trim();
        }
        // Uppercase snapshot for suffix detection
        let v_upper = v_unparen.to_uppercase();

        // If it's a quoted string, it's a string constant
        if v_unparen.starts_with('"') && v_unparen.ends_with('"') {
            return "&'static str".to_string();
        }

        // Detect explicit C-style unsigned suffixes
        let is_ull = v_upper.ends_with("ULL");
        // ends_with with a single char is fine here; guard against matching U in ULL
        let is_u = !is_ull && v_upper.ends_with('U');

        if is_ull {
            return "u64".to_string();
        }
        if is_u {
            return "u32".to_string();
        }

        // Handle unary bitwise operators (e.g., ~0U) as numeric values
        if v_unparen.starts_with('~') {
            // If no explicit suffix was present, default to u32 for common Vulkan constants
            return "u32".to_string();
        }

        if v_unparen.starts_with("0x") || v_unparen.starts_with("0X") {
            // Unsuffixed hex — the explicit `U`/`ULL` suffixes were handled
            // above, so the width has to come from the magnitude. Anything
            // that fits is `u32` (the overwhelming majority: structure types,
            // format numbers, version words); wider literals get `u64` so
            // 64-bit values such as extended flag bits don't emit as a `u32`
            // constant the compiler then rejects as out of range.
            match u128::from_str_radix(&v_unparen[2..].replace('_', ""), 16) {
                Ok(n) if n > u32::MAX as u128 => "u64".to_string(),
                // Unparseable digits keep the historical default rather than
                // guessing wider; a genuinely malformed literal is a build
                // error either way.
                _ => "u32".to_string(),
            }
        } else if v.contains('.') {
            // Float values
            "f32".to_string()
        } else if v_unparen.parse::<i64>().is_ok() {
            // Integer values - assume u32 for non-negative, i32 for negative
            if v_unparen.starts_with('-') {
                "i32".to_string()
            } else {
                "u32".to_string()
            }
        } else {
            // Fallback to string-like value
            "&'static str".to_string()
        }
    }

    /// Map Vulkan values to Rust values
    fn map_value(&self, value: &str, value_type: &str) -> String {
        // Handle hex values
        let v = value.trim();
        // strip repeated surrounding parentheses for mapping
        let mut v_unparen = v;
        while v_unparen.starts_with('(') && v_unparen.ends_with(')') {
            v_unparen = v_unparen.trim_start_matches('(').trim_end_matches(')');
            v_unparen = v_unparen.trim();
        }

        if v_unparen.starts_with("0x") || v_unparen.starts_with("0X") {
            return v_unparen.to_string();
        }

        // Handle string values (rare for constants)
        if v_unparen.starts_with('"') && v_unparen.ends_with('"') {
            return v_unparen.to_string();
        }

        // Handle float values (value_type may be "f32"/"f64" or "float")
        if (value_type.starts_with('f') || value_type == "float") && !value.contains('.') {
            return format!("{}.0", value);
        }

        // Detect unsigned suffix context before transformations (only trailing)
        let v_upper_unparen = v_unparen.to_uppercase();
        let is_ull = v_upper_unparen.ends_with("ULL");
        let is_u = v_upper_unparen.ends_with('U') && !is_ull;

        // Handle bitwise NOT operator ~ -> ! and preserve suffix typing
        if v_unparen.starts_with('~') {
            // Replace leading ~ with Rust !
            let mut replaced = v_unparen.replacen('~', "!", 1);

            // Remove any existing C-style unsigned suffixes (U, UL, ULL)
            // before appending Rust-style suffixes to avoid producing invalid
            // literals like `!0Uu32`.
            let mut rep_upper = replaced.to_uppercase();
            if rep_upper.ends_with("ULL") {
                replaced.truncate(replaced.len() - 3);
                rep_upper.truncate(rep_upper.len() - 3);
            } else if rep_upper.ends_with('U') {
                replaced.truncate(replaced.len() - 1);
                rep_upper.truncate(rep_upper.len() - 1);
            }

            if is_ull {
                return format!("{}u64", replaced);
            } else if is_u {
                return format!("{}u32", replaced);
            } else {
                return replaced;
            }
        }

        // Handle C-style literals with proper suffix matching
        let mut rust_value = v_unparen.to_string();

        // Replace suffixes at the end of numbers
        if rust_value.to_uppercase().ends_with("ULL") {
            // strip ULL suffix for numeric literal, we'll use Rust forms when needed
            rust_value = rust_value[..rust_value.len() - 3].to_string();
        }
        if rust_value.ends_with("U)") {
            rust_value = rust_value.replace("U)", ")");
        }
        if rust_value.ends_with("U;") {
            rust_value = rust_value.replace("U;", ";");
        }
        if rust_value.ends_with("F") || rust_value.ends_with("f") {
            // Replace only trailing F or f with f32
            rust_value = rust_value.trim_end_matches(['F', 'f']).to_string() + "f32";
        }
        if rust_value.ends_with("L") {
            rust_value = rust_value.replace("L", "");
        }

        // Handle C bitwise NOT (~0) - convert to Rust equivalent
        // Handle common parenthesized bitwise NOT patterns
        if rust_value == "~0ULL" || rust_value == "(~0ULL)" {
            rust_value = "u64::MAX".to_string();
        } else if rust_value == "~0U" || rust_value == "(~0U)" {
            rust_value = "!0u32".to_string();
        } else if rust_value == "~0" || rust_value == "(~0)" {
            rust_value = "!0".to_string();
        } else if rust_value == "~1" || rust_value == "(~1)" {
            rust_value = "!1".to_string();
        } else if rust_value == "~2" || rust_value == "(~2)" {
            rust_value = "!2".to_string();
        }

        rust_value
    }
}

impl GeneratorModule for ConstantGenerator {
    fn name(&self) -> &str {
        "ConstantGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["constants.json".to_string()]
    }

    fn output_file(&self) -> String {
        "constants.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        Vec::new() // Constants don't depend on other modules
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("constants.json");
        let input_content = fs::read_to_string(input_path).map_err(GeneratorError::Io)?;

        // Parse JSON as simple array of constants
        let constants_array: Vec<ConstantDefinition> =
            serde_json::from_str(&input_content).map_err(GeneratorError::Json)?;

        // Generate code
        let mut generated_code = String::new();
        let mut seen_constants = std::collections::HashSet::new();

        // Generate constants with deduplication
        for constant in &constants_array {
            // Skip duplicate constants
            if seen_constants.contains(&constant.name) {
                continue;
            }
            seen_constants.insert(constant.name.clone());

            generated_code.push_str(&self.generate_constant(constant));
            generated_code.push('\n');
        }

        // Write output file
        let output_path = output_dir.join(self.output_file());
        fs::write(output_path, generated_code).map_err(GeneratorError::Io)?;

        log_info(&format!("Generated {} constants", constants_array.len()));
        Ok(())
    }

    /// Constants define named values, not types, so both type lists are empty
    /// here and that is the accurate answer rather than a placeholder.
    ///
    /// `used_types` stays empty for a second reason: the intermediate
    /// `ConstantDefinition` carries no declared C type — the Rust type is
    /// recovered from the literal by `infer_type_from_value` — so there is no
    /// type reference to report. Informational either way; see
    /// [`GeneratorMetadata`].
    fn metadata(&self) -> GeneratorMetadata {
        GeneratorMetadata {
            defined_types: Vec::new(),
            used_types: Vec::new(),
            has_forward_declarations: false,
            priority: 10, // Constants should be generated early
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_generation() {
        let generator = ConstantGenerator::new();

        let constant = ConstantDefinition {
            name: "VK_API_VERSION_1_0".to_string(),
            value: Some("0x100000".to_string()),
            source_line: None,
            alias: None,
            comment: None,
            api: None,
            deprecated: None,
            constant_type: "enum".to_string(),
            raw_content: String::new(),
            is_alias: false,
        };

        let generated = generator.generate_constant(&constant);

        assert!(generated.contains("pub const VK_API_VERSION_1_0: u32 = 0x100000;"));
        // Note: No source line info in simplified format
        // assert!(generated.contains("from line 42"));
    }

    #[test]
    fn test_value_mapping() {
        let generator = ConstantGenerator::new();

        assert_eq!(generator.map_value("0x1000", "uint32_t"), "0x1000");
        assert_eq!(generator.map_value("42", "float"), "42.0");
        assert_eq!(generator.map_value("123", "uint32_t"), "123");
    }

    /// An unsuffixed hex literal has to be typed by magnitude. Typing a
    /// value wider than 32 bits as `u32` emits a constant the compiler
    /// rejects as out of range, breaking the whole generated bindings file.
    #[test]
    fn unsuffixed_hex_is_typed_by_magnitude() {
        let generator = ConstantGenerator::new();

        // The common case: everything that fits stays u32.
        assert_eq!(generator.infer_type_from_value("0x1000"), "u32");
        assert_eq!(generator.infer_type_from_value("0xFFFFFFFF"), "u32");
        assert_eq!(generator.infer_type_from_value("0x0"), "u32");

        // One past u32::MAX, and beyond.
        assert_eq!(generator.infer_type_from_value("0x100000000"), "u64");
        assert_eq!(generator.infer_type_from_value("0xFFFFFFFFFFFFFFFF"), "u64");

        // Underscore separators must not change the verdict.
        assert_eq!(generator.infer_type_from_value("0x1_0000_0000"), "u64");
        assert_eq!(generator.infer_type_from_value("0xFFFF_FFFF"), "u32");

        // Uppercase prefix takes the same path.
        assert_eq!(generator.infer_type_from_value("0X100000000"), "u64");

        // Explicit C suffixes still win over magnitude — they are the
        // author's stated intent and are handled before this branch.
        assert_eq!(generator.infer_type_from_value("0x1000U"), "u32");
        assert_eq!(generator.infer_type_from_value("0x1000ULL"), "u64");
    }
}
