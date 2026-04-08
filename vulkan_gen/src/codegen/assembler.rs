//! Code assembler for Vulkan bindings
//!
//! This module handles the final assembly of all generated code fragments
//! into a single, dependency-ordered bindings.rs file.

use crate::codegen::logging::{log_info, log_warn};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use thiserror::Error;

use super::generator_modules::{CodeFragment, GeneratorModule};

// Simple helper: extract top-level defined type names from a code fragment
fn extract_defined_type_names(code: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in code.lines() {
        let l = line.trim_start();
        if l.starts_with("pub struct ") {
            if let Some(rest) = l.split_whitespace().nth(2) {
                // Remove trailing brace or generics
                let name = rest
                    .trim_end_matches('{')
                    .trim_end_matches('<')
                    .trim()
                    .to_string();
                names.push(name);
            }
        } else if l.starts_with("pub enum ") {
            if let Some(rest) = l.split_whitespace().nth(2) {
                let name = rest.trim_end_matches('{').trim().to_string();
                names.push(name);
            }
        } else if l.starts_with("pub type ") {
            if let Some(rest) = l.split_whitespace().nth(2) {
                let name = rest.trim_end_matches('=').trim().to_string();
                names.push(name);
            }
        }
    }
    names
}

// Remove a top-level type definition block (struct/enum/type) for `name` from code
fn remove_type_definition(code: &str, name: &str) -> String {
    let mut out = String::new();
    let mut skip = false;
    let mut brace_depth = 0usize;

    for line in code.lines() {
        if !skip {
            let trimmed = line.trim_start();
            if (trimmed.starts_with("pub struct ") || trimmed.starts_with("pub enum "))
                && trimmed.contains(name)
            {
                // start skipping until matching brace depth returns to 0
                skip = true;
                // Count braces on this line
                brace_depth = trimmed.matches('{').count() - trimmed.matches('}').count();
                if brace_depth == 0 {
                    // single-line struct/enum; continue skipping this line only
                    skip = false;
                    continue;
                }
                continue;
            }
            if trimmed.starts_with("pub type ") && trimmed.contains(name) {
                // pub type Alias = ...; skip this single line
                continue;
            }
            out.push_str(line);
            out.push('\n');
        } else {
            // currently skipping a block
            brace_depth += line.matches('{').count();
            brace_depth = brace_depth.saturating_sub(line.matches('}').count());
            if brace_depth == 0 {
                skip = false;
            }
            // continue skipping
        }
    }

    out
}

/// Error type for assembler operations
#[derive(Debug, Error)]
pub enum AssemblerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Generator error: {0}")]
    Generator(#[from] super::generator_modules::GeneratorError),

    #[error("Circular dependency detected: {cycle:?}")]
    CircularDependency { cycle: Vec<String> },

    #[error("Missing dependency: {dependency} required by {module}")]
    MissingDependency { dependency: String, module: String },

    #[error("Code validation failed: {message}")]
    Validation { message: String },

    #[error("Duplicate type definition: {type_name} defined in multiple modules")]
    DuplicateType { type_name: String },
}

/// Result type for assembler operations
pub type AssemblerResult<T> = Result<T, AssemblerError>;

/// The main code assembler
pub struct CodeAssembler {
    modules: Vec<Box<dyn GeneratorModule>>,
    fragments: HashMap<String, CodeFragment>,
    dependency_graph: HashMap<String, Vec<String>>,
}

impl CodeAssembler {
    /// Create a new code assembler
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            fragments: HashMap::new(),
            dependency_graph: HashMap::new(),
        }
    }

    /// Register a generator module
    pub fn register_module(&mut self, module: Box<dyn GeneratorModule>) {
        let dependencies = module.dependencies();
        let name = module.name().to_string();

        self.dependency_graph.insert(name, dependencies);
        self.modules.push(module);
    }

    /// Generate all code fragments from intermediate files
    pub fn generate_fragments(
        &mut self,
        input_dir: &Path,
        output_dir: &Path,
    ) -> AssemblerResult<()> {
        // Create output directory
        fs::create_dir_all(output_dir)?;

        // Get execution order based on dependencies
        let execution_order = self.resolve_dependencies()?;

        // Generate fragments in dependency order
        for module_name in &execution_order {
            let module_name_str: &str = module_name.as_str();
            if let Some(module) = self.modules.iter().find(|m| m.name() == module_name_str) {
                log_info(&format!("Generating code with {}", module.name()));

                // Generate code with enhanced error reporting
                if let Err(e) = module.generate(input_dir, output_dir) {
                    eprintln!("Error in generator module '{}': {}", module.name(), e);
                    return Err(AssemblerError::Generator(e));
                }

                // Read generated file and create fragment
                let output_file = output_dir.join(module.output_file());
                if output_file.exists() {
                    let code = fs::read_to_string(&output_file)?;
                    let fragment =
                        CodeFragment::new(code, module.name()).with_metadata(module.metadata());

                    self.fragments.insert(module.name().to_string(), fragment);
                }
            }
        }

        Ok(())
    }

    /// Assemble all fragments into final bindings.rs
    pub fn assemble_final_bindings(&self, output_path: &Path) -> AssemblerResult<()> {
        let mut final_code = String::new();

        // File header comment
        final_code.push_str("// Vulkan bindings\n");
        final_code.push_str("//\n");
        final_code
            .push_str("// This file is automatically generated by the event-driven parser.\n");
        final_code.push_str("// Do not edit manually.\n\n");

        // Note: c_void and c_char are imported by the enclosing module (raw/mod.rs)
        // so we don't need to import them here.

        // Get assembly order (same as dependency order)
        let assembly_order = self.resolve_dependencies()?;

        // Track which types we've already emitted so we can deduplicate
        let mut emitted_types: HashSet<String> = HashSet::new();

        // Assemble fragments in order
        for module_name in &assembly_order {
            if let Some(fragment) = self.fragments.get(module_name.as_str()) {
                final_code.push_str(&format!("// === {} ===\n", module_name));

                // Determine types defined in this fragment. Prefer explicit metadata,
                // otherwise fallback to scanning the code for `pub struct/enum/type`.
                let mut defined_names: Vec<String> = Vec::new();
                if !fragment.metadata.defined_types.is_empty() {
                    defined_names.extend(fragment.metadata.defined_types.clone());
                } else {
                    defined_names.extend(extract_defined_type_names(&fragment.code));
                }

                // Start with the raw fragment code and remove any duplicate type
                // definitions that were already emitted earlier.
                let mut code_to_append = fragment.code.clone();
                for name in &defined_names {
                    if emitted_types.contains(name) {
                        // Remove the type/enum block from the fragment before appending
                        code_to_append = remove_type_definition(&code_to_append, name);
                        // Insert a user-friendly comment so the generated file documents
                        // that a duplicate definition was intentionally skipped.
                        code_to_append = format!(
                            "{}// Skipped other alias for {} because an enum or struct with that name exists\n",
                            code_to_append, name
                        );
                    } else {
                        emitted_types.insert(name.clone());
                    }
                }

                final_code.push_str(&code_to_append);
                final_code.push('\n');
            }
        }

        // Write final file. Attempt direct write first; on PermissionDenied fall back to
        // writing to a temporary file in the same directory and renaming it into place.
        match fs::write(output_path, final_code.clone()) {
            Ok(_) => {}
            Err(e) => {
                use std::io::ErrorKind;
                if e.kind() == ErrorKind::PermissionDenied {
                    // Attempt atomic write via temp file in same directory
                    if let Some(parent) = output_path.parent() {
                        // Build a unique temp file name
                        let mut attempt = 0u8;
                        let mut last_err: Option<std::io::Error> = None;
                        while attempt < 5 {
                            let tmp_name = format!(".vulkan_bindings.rs.tmp.{}", attempt);
                            let tmp_path = parent.join(&tmp_name);
                            // Try to create and write the temp file
                            match fs::write(&tmp_path, final_code.as_bytes()) {
                                Ok(_) => {
                                    // Try to atomically rename into place
                                    match std::fs::rename(&tmp_path, output_path) {
                                        Ok(_) => {
                                            last_err = None;
                                            break;
                                        }
                                        Err(rename_err) => {
                                            last_err = Some(rename_err);
                                            // Attempt to remove tmp and retry
                                            let _ = std::fs::remove_file(&tmp_path);
                                        }
                                    }
                                }
                                Err(write_tmp_err) => {
                                    last_err = Some(write_tmp_err);
                                }
                            }

                            attempt += 1;
                            // Small sleep to allow transient locks to clear
                            std::thread::sleep(std::time::Duration::from_millis(50));
                        }

                        if let Some(err_final) = last_err {
                            return Err(AssemblerError::Io(err_final));
                        }
                    } else {
                        return Err(AssemblerError::Io(e));
                    }
                } else {
                    return Err(AssemblerError::Io(e));
                }
            }
        }

        log_info(&format!(
            "Assembled final bindings.rs with {} modules",
            self.fragments.len()
        ));
        Ok(())
    }

    /// Resolve module dependencies using topological sort
    fn resolve_dependencies(&self) -> AssemblerResult<Vec<String>> {
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();
        let mut result = Vec::new();

        // Get all module names
        let module_names: HashSet<String> = self.dependency_graph.keys().cloned().collect();

        // Perform topological sort
        for module in &module_names {
            if !visited.contains(module) {
                self.dfs_visit(module, &mut visited, &mut temp_visited, &mut result)?;
            }
        }

        // `result` is built by pushing a module after its dependencies
        // (post-order), so it already represents a valid topological
        // ordering where dependencies come before dependents.
        Ok(result)
    }

    /// Depth-first search for topological sort
    fn dfs_visit(
        &self,
        module: &str,
        visited: &mut HashSet<String>,
        temp_visited: &mut HashSet<String>,
        result: &mut Vec<String>,
    ) -> AssemblerResult<()> {
        if temp_visited.contains(module) {
            return Err(AssemblerError::CircularDependency {
                cycle: vec![module.to_string()],
            });
        }

        if visited.contains(module) {
            return Ok(());
        }

        temp_visited.insert(module.to_string());

        // Visit dependencies first
        if let Some(dependencies) = self.dependency_graph.get(module) {
            for dep in dependencies {
                if !self.dependency_graph.contains_key(dep) {
                    return Err(AssemblerError::MissingDependency {
                        dependency: dep.clone(),
                        module: module.to_string(),
                    });
                }
                self.dfs_visit(dep, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(module);
        visited.insert(module.to_string());
        result.push(module.to_string());

        Ok(())
    }

    /// Validate the generated code for common issues
    pub fn validate_generated_code(&self) -> AssemblerResult<()> {
        let mut defined_types = HashMap::new();

        // Check for duplicate type definitions
        for (module_name, fragment) in &self.fragments {
            for defined_type in &fragment.metadata.defined_types {
                if let Some(_existing_module) =
                    defined_types.insert(defined_type.clone(), module_name.clone())
                {
                    return Err(AssemblerError::DuplicateType {
                        type_name: defined_type.clone(),
                    });
                }
            }
        }

        // Check that all used types are defined
        for (module_name, fragment) in &self.fragments {
            for used_type in &fragment.metadata.used_types {
                if !defined_types.contains_key(used_type) {
                    log_warn(&format!(
                        "Module {} uses undefined type: {}",
                        module_name, used_type
                    ));
                }
            }
        }

        Ok(())
    }
}

impl Default for CodeAssembler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_resolution() {
        let mut assembler = CodeAssembler::new();

        // Create mock modules with dependencies
        assembler
            .dependency_graph
            .insert("constants".to_string(), vec![]);
        assembler
            .dependency_graph
            .insert("enums".to_string(), vec![]);
        assembler.dependency_graph.insert(
            "structs".to_string(),
            vec!["enums".to_string(), "constants".to_string()],
        );
        assembler
            .dependency_graph
            .insert("functions".to_string(), vec!["structs".to_string()]);

        let order = assembler.resolve_dependencies().unwrap();

        // Constants and enums should come first (order between them doesn't matter)
        // Structs should come after both
        // Functions should come last
        let constants_pos = order.iter().position(|x| x == "constants").unwrap();
        let enums_pos = order.iter().position(|x| x == "enums").unwrap();
        let structs_pos = order.iter().position(|x| x == "structs").unwrap();
        let functions_pos = order.iter().position(|x| x == "functions").unwrap();

        assert!(constants_pos < structs_pos);
        assert!(enums_pos < structs_pos);
        assert!(structs_pos < functions_pos);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut assembler = CodeAssembler::new();

        // Create circular dependency
        assembler
            .dependency_graph
            .insert("a".to_string(), vec!["b".to_string()]);
        assembler
            .dependency_graph
            .insert("b".to_string(), vec!["a".to_string()]);

        let result = assembler.resolve_dependencies();
        assert!(matches!(
            result,
            Err(AssemblerError::CircularDependency { .. })
        ));
    }
}
