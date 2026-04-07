//! Function generator module
//!
//! Generates Rust function signatures from functions.json intermediate file

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorModule, GeneratorResult};

use crate::parser::vk_types::FunctionDefinition;

/// Generator module for Vulkan functions
pub struct FunctionGenerator;

impl FunctionGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Rust code for a single function
    fn generate_function(&self, func_def: &FunctionDefinition) -> String {
        // Build an atomic emission for the typedef so sanitizer cannot see
        // fragmented parts. If we don't have parameters or a sensible
        // return type, emit a safe pointer alias fallback which is valid
        // Rust and easy to understand in diagnostics.
        let mut code = String::new();

        // Documentation comment from vk.xml if present, otherwise the function name
        if let Some(comment) = &func_def.comment {
            for line in comment.lines() {
                code.push_str(&format!("/// {}\n", line.trim()));
            }
        } else {
            code.push_str(&format!("/// Vulkan function: `{}`\n", func_def.name));
        }

        // If we don't have parameter information, avoid emitting a broken
        // function-pointer typedef; emit a pointer alias instead.
        if func_def.parameters.is_empty() {
            let alias = format!("pub type {} = *mut c_void;\n\n", func_def.name);
            code.push_str(&alias);
            return code;
        }

        // Otherwise, emit a full function-pointer typedef atomically.
        let mut sig = String::new();
        sig.push_str(&format!(
            "pub type {} = unsafe extern \"system\" fn(",
            func_def.name
        ));

        let mut params = Vec::new();
        for param in &func_def.parameters {
            let rust_type = self.map_param_type_from_definition(param);
            let param_name = self.escape_rust_keyword(&param.name);
            params.push(format!("{}: {}", param_name, rust_type));
        }

        sig.push_str(&params.join(", "));

        // Return type mapping. C `void` becomes Rust `()`, not `c_void`
        // (which is an opaque type, not a unit return value).
        let raw_ret = func_def.return_type.trim();
        let return_type = if raw_ret.is_empty() || raw_ret == "void" {
            "()".to_string()
        } else {
            match raw_ret {
                "const" | "fn" => "*mut c_void".to_string(),
                other => {
                    let mapped = self.simple_map_param_type(other);
                    if mapped.is_empty() || mapped == "c_void" {
                        "()".to_string()
                    } else {
                        mapped
                    }
                }
            }
        };

        sig.push_str(&format!(") -> {};\n\n", return_type));

        code.push_str(&sig);
        code
    }

    /// Simple type mapping for simplified intermediate types
    fn simple_map_param_type(&self, type_name: &str) -> String {
        match type_name {
            "void" => "c_void".to_string(),
            "char" => "c_char".to_string(),
            "uint8_t" => "u8".to_string(),
            "uint16_t" => "u16".to_string(),
            "uint32_t" => "u32".to_string(),
            "uint64_t" => "u64".to_string(),
            "int8_t" => "i8".to_string(),
            "int16_t" => "i16".to_string(),
            "int32_t" => "i32".to_string(),
            "int64_t" => "i64".to_string(),
            "float" => "f32".to_string(),
            "double" => "f64".to_string(),
            "size_t" => "usize".to_string(),
            "int" => "i32".to_string(),
            "unsigned" => "u32".to_string(),
            _ => {
                if type_name.trim().is_empty() {
                    // Unknown type - fall back to c_void so emitted Rust is valid
                    "c_void".to_string()
                } else {
                    type_name.to_string()
                }
            } // Keep Vulkan types as-is for now
        }
    }

    /// Map a parameter to a Rust type using the C-style `definition` string if present.
    fn map_param_type_from_definition(
        &self,
        param: &crate::parser::vk_types::FunctionParameter,
    ) -> String {
        // If we have a C-style definition like "const VkInstanceCreateInfo* pCreateInfo",
        // inspect it for pointer asterisks and a `const` qualifier. Fall back to the
        // simple mapping of the base type otherwise.
        let def = param.definition.trim();

        // Count pointer level by counting '*' characters
        let pointer_level = def.chars().filter(|c| *c == '*').count();

        // Determine if the type is const-qualified (appears before the base type)
        let const_qualified = def.contains("const");

        // Start from the base Rust-mapped type. If the param type is missing
        // fall back to c_void so we don't emit empty types.
        let mut rust = self.simple_map_param_type(&param.type_name);
        if rust.trim().is_empty() {
            rust = "c_void".to_string();
        }

        if pointer_level == 0 {
            return rust;
        }

        // Apply pointer wrapping: outermost pointer uses const qualification when present
        for i in 0..pointer_level {
            if i == 0 {
                if const_qualified {
                    rust = format!("*const {}", rust);
                } else {
                    rust = format!("*mut {}", rust);
                }
            } else {
                // Inner pointers default to mutable pointers
                rust = format!("*mut {}", rust);
            }
        }

        rust
    }

    fn escape_rust_keyword(&self, name: &str) -> String {
        match name {
            "type" | "match" | "impl" | "fn" | "let" | "mut" | "const" | "static" | "if"
            | "else" | "while" | "for" | "loop" | "break" | "continue" | "return" | "struct"
            | "enum" | "trait" | "mod" | "pub" | "use" | "extern" | "crate" | "self" | "Self"
            | "super" | "where" | "async" | "await" | "dyn" | "abstract" | "become" | "box"
            | "do" | "final" | "macro" | "override" | "priv" | "typeof" | "unsized" | "virtual"
            | "yield" | "try" | "union" | "ref" => format!("r#{}", name),
            _ => name.to_string(),
        }
    }

    /// Classify a command as entry-level, instance-level, or device-level
    /// based on the type of its first parameter.
    fn classify_command(func: &FunctionDefinition) -> &'static str {
        if func.is_alias || func.parameters.is_empty() {
            return "entry";
        }
        let first_type = &func.parameters[0].type_name;
        match first_type.as_str() {
            "VkDevice" | "VkCommandBuffer" | "VkQueue" => "device",
            "VkInstance" | "VkPhysicalDevice" => "instance",
            _ => "entry",
        }
    }

    /// Generate dispatch table structs for entry, instance, and device commands.
    fn generate_dispatch_tables(&self, functions: &[FunctionDefinition]) -> String {
        let mut entry_fns = Vec::new();
        let mut instance_fns = Vec::new();
        let mut device_fns = Vec::new();

        for func in functions {
            if func.is_alias || func.parameters.is_empty() && func.return_type.is_empty() {
                continue;
            }
            // Skip alias commands (they share the same function pointer)
            if func.alias.is_some() {
                continue;
            }

            match Self::classify_command(func) {
                "entry" => entry_fns.push(func),
                "instance" => instance_fns.push(func),
                "device" => device_fns.push(func),
                _ => {}
            }
        }

        let mut code = String::new();

        // Entry dispatch table (global functions loaded without an instance)
        code.push_str("/// Global Vulkan functions loaded without an instance\n");
        code.push_str("#[allow(non_snake_case)]\n");
        code.push_str("pub struct VkEntryDispatchTable {\n");
        for func in &entry_fns {
            code.push_str(&format!("    pub {}: Option<{}>,\n", func.name, func.name));
        }
        code.push_str("}\n\n");

        // Instance dispatch table
        code.push_str("/// Instance-level Vulkan functions\n");
        code.push_str("#[allow(non_snake_case)]\n");
        code.push_str("pub struct VkInstanceDispatchTable {\n");
        for func in &instance_fns {
            code.push_str(&format!("    pub {}: Option<{}>,\n", func.name, func.name));
        }
        code.push_str("}\n\n");

        // Device dispatch table
        code.push_str("/// Device-level Vulkan functions\n");
        code.push_str("#[allow(non_snake_case)]\n");
        code.push_str("pub struct VkDeviceDispatchTable {\n");
        for func in &device_fns {
            code.push_str(&format!("    pub {}: Option<{}>,\n", func.name, func.name));
        }
        code.push_str("}\n\n");

        // Generate load functions for each table
        code.push_str(&Self::generate_load_fn(
            "VkEntryDispatchTable",
            &entry_fns,
            "null_mut",
        ));
        code.push_str(&Self::generate_load_fn(
            "VkInstanceDispatchTable",
            &instance_fns,
            "instance",
        ));
        code.push_str(&Self::generate_load_fn(
            "VkDeviceDispatchTable",
            &device_fns,
            "device",
        ));

        code
    }

    fn generate_load_fn(
        table_name: &str,
        functions: &[&FunctionDefinition],
        _context: &str,
    ) -> String {
        let mut code = String::new();
        code.push_str(&format!("impl {} {{\n", table_name));
        code.push_str("    /// Load all function pointers using the provided loader function.\n");
        code.push_str("    /// `load_fn` takes a function name and returns a raw pointer.\n");
        code.push_str("    pub unsafe fn load(load_fn: impl Fn(&std::ffi::CStr) -> *mut std::ffi::c_void) -> Self {\n");
        code.push_str("      unsafe {\n");
        code.push_str("        Self {\n");
        for func in functions {
            code.push_str(&format!(
                "            {name}: {{\n\
                 \x20               let ptr = load_fn(c\"{name}\");\n\
                 \x20               if ptr.is_null() {{ None }} else {{ Some(std::mem::transmute(ptr)) }}\n\
                 \x20           }},\n",
                name = func.name
            ));
        }
        code.push_str("        }\n");
        code.push_str("      }\n");
        code.push_str("    }\n");
        code.push_str("}\n\n");
        code
    }
}

impl GeneratorModule for FunctionGenerator {
    fn name(&self) -> &str {
        "FunctionGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["functions.json".to_string()]
    }

    fn output_file(&self) -> String {
        "functions.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        vec![
            "StructGenerator".to_string(),
            "EnumGenerator".to_string(),
            "ConstantGenerator".to_string(),
        ]
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        let mut generated_code = String::new();

        // First, emit funcpointer types from types.json (they reference struct types
        // so they must come after structs, which is why they're here instead of in TypeGenerator)
        let types_path = input_dir.join("types.json");
        if types_path.exists() {
            let types_content =
                fs::read_to_string(&types_path).map_err(|e| GeneratorError::Io(e))?;
            if let Ok(types) =
                serde_json::from_str::<Vec<crate::parser::vk_types::TypeDefinition>>(&types_content)
            {
                let type_gen = crate::generator_modules::type_gen::TypeGenerator::new();
                let all_type_names = std::collections::HashSet::new();
                generated_code.push_str("// Function pointer types\n");
                for type_def in types.iter().filter(|t| t.category == "funcpointer") {
                    generated_code.push_str(&type_gen.generate_type_public(
                        type_def,
                        &all_type_names,
                        output_dir,
                    ));
                }
                generated_code.push('\n');
            }
        }

        // Read function definitions
        let input_path = input_dir.join("functions.json");
        let input_content = fs::read_to_string(input_path).map_err(|e| GeneratorError::Io(e))?;

        let functions: Vec<FunctionDefinition> =
            serde_json::from_str(&input_content).map_err(|e| GeneratorError::Json(e))?;

        // Generate command function pointer typedefs
        for func_def in &functions {
            generated_code.push_str(&self.generate_function(func_def));
        }

        // Generate dispatch tables
        generated_code.push_str(&self.generate_dispatch_tables(&functions));

        // Write output file
        let output_path = output_dir.join(self.output_file());
        fs::write(output_path, generated_code).map_err(|e| GeneratorError::Io(e))?;

        crate::codegen::logging::log_info(&format!(
            "FunctionGeneratorModule: Generated {} function signatures",
            functions.len()
        ));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_function_generation() {
        use crate::parser::vk_types::{FunctionDefinition, FunctionParameter};

        let generator = FunctionGenerator::new();

        let func_def = FunctionDefinition {
            name: "vkCreateInstance".to_string(),
            return_type: "VkResult".to_string(),
            comment: Some("Create a new Vulkan instance".to_string()),
            successcodes: None,
            errorcodes: None,
            alias: None,
            api: None,
            deprecated: None,
            cmdbufferlevel: None,
            pipeline: None,
            queues: None,
            renderpass: None,
            videocoding: None,
            parameters: vec![FunctionParameter {
                name: "pCreateInfo".to_string(),
                type_name: "VkInstanceCreateInfo".to_string(),
                optional: None,
                len: None,
                altlen: None,
                externsync: None,
                noautovalidity: None,
                objecttype: None,
                stride: None,
                validstructs: None,
                api: None,
                deprecated: None,
                comment: None,
                definition: "const VkInstanceCreateInfo* pCreateInfo".to_string(),
                raw_content: String::new(),
                source_line: None,
            }],
            raw_content: String::new(),
            is_alias: false,
            source_line: None,
        };

        let code = generator.generate_function(&func_def);

        assert!(code.contains("pub type vkCreateInstance"));
        assert!(code.contains("pCreateInfo: *const VkInstanceCreateInfo"));
        assert!(code.contains("-> VkResult"));
    }
}
