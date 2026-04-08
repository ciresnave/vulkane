//! Integration module for shared intermediate types
//!
//! This module ensures that all parsing and generation modules use the same
//! shared data structures for consistent code generation. It provides validation
//! to ensure type consistency and compatibility across the entire system.

use std::collections::{HashMap, HashSet};
use std::io::Error as IoError;
use std::path::Path;
use thiserror::Error;

use crate::codegen::logging::{log_debug, log_error, log_info};
use crate::parser::vk_types::*;

/// Helper function to determine if a type is primitive
fn is_primitive_type(type_name: &str) -> bool {
    matches!(
        type_name,
        "void"
            | "char"
            | "uint8_t"
            | "uint16_t"
            | "uint32_t"
            | "uint64_t"
            | "int32_t"
            | "int64_t"
            | "float"
            | "double"
            | "size_t"
    )
}

/// Helper function to determine if a type is an opaque handle
fn is_opaque_handle(type_name: &str) -> bool {
    type_name.starts_with("Vk")
        && (type_name.contains("Handle")
            || matches!(
                type_name,
                "VkInstance"
                    | "VkPhysicalDevice"
                    | "VkDevice"
                    | "VkQueue"
                    | "VkCommandBuffer"
                    | "VkBuffer"
                    | "VkImage"
                    | "VkSemaphore"
                    | "VkFence"
                    | "VkDeviceMemory"
                    | "VkEvent"
                    | "VkQueryPool"
                    | "VkBufferView"
                    | "VkImageView"
                    | "VkShaderModule"
                    | "VkPipelineCache"
                    | "VkPipelineLayout"
                    | "VkRenderPass"
                    | "VkPipeline"
                    | "VkDescriptorSetLayout"
                    | "VkSampler"
                    | "VkDescriptorPool"
                    | "VkDescriptorSet"
                    | "VkFramebuffer"
                    | "VkCommandPool"
                    | "VkSurfaceKHR"
            ))
}

/// Result type for type integration operations
pub type TypeIntegrationResult<T> = Result<T, TypeIntegrationError>;

/// Error type for the type integration system
#[derive(Debug, Error)]
pub enum TypeIntegrationError {
    #[error("IO error: {0}")]
    Io(#[from] IoError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Duplicate type definition: {type_name} defined in {locations:?}")]
    DuplicateType {
        type_name: String,
        locations: Vec<String>,
    },

    #[error("Type not found: {type_name} referenced in {referenced_in}")]
    TypeNotFound {
        type_name: String,
        referenced_in: String,
    },

    #[error("Type mismatch: {type_name} defined as {defined_as} but used as {used_as}")]
    TypeMismatch {
        type_name: String,
        defined_as: String,
        used_as: String,
    },

    #[error("Circular dependency detected: {0:?}")]
    CircularDependency(Vec<String>),

    #[error("Inconsistent pointer usage: {type_name} used with inconsistent pointer levels")]
    InconsistentPointerUsage { type_name: String },

    #[error("Validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("General validation error: {0}")]
    ValidationError(String),
}

/// Verify that all required intermediate files exist
pub fn verify_intermediate_files(intermediate_dir: &Path) -> TypeIntegrationResult<()> {
    let required_files = [
        "structs.json",
        "enums.json",
        "constants.json",
        "functions.json",
        "types.json",
        "extensions.json",
        "macros.json",
        "includes.json",
    ];

    log_info("Verifying intermediate files...");

    let mut missing_files = Vec::new();

    for file in &required_files {
        let file_path = intermediate_dir.join(file);
        if !file_path.exists() {
            log_error(&format!("Required intermediate file not found: {}", file));
            missing_files.push(file.to_string());
        }
    }

    if !missing_files.is_empty() {
        return Err(TypeIntegrationError::ValidationError(format!(
            "Missing required intermediate files: {:?}",
            missing_files
        )));
    }

    log_info("All required intermediate files verified");
    Ok(())
}

/// Perform comprehensive consistency checks on intermediate data
pub fn check_data_consistency(intermediate_dir: &Path) -> TypeIntegrationResult<()> {
    log_info("Performing comprehensive data consistency checks across intermediate files...");

    // Read all intermediate files with better error reporting
    let structs_path = intermediate_dir.join("structs.json");
    log_info(&format!("Reading structs.json from {:?}", structs_path));
    let structs_content = std::fs::read_to_string(&structs_path)?;
    // Accept either a plain JSON array of StructDefinition or the legacy
    // object-with-array format { "structs": [...] }.
    // Be tolerant of legacy JSON shapes: older output used "fields" and
    // "field_type" keys. Try parsing directly, and if that fails attempt to
    // normalize common legacy keys to the current shape and parse again.
    let structs_data: StructData =
        match serde_json::from_str::<Vec<StructDefinition>>(&structs_content) {
            Ok(vec) => StructData { structs: vec },
            Err(_) => match serde_json::from_str::<StructData>(&structs_content) {
                Ok(sd) => sd,
                Err(_) => {
                    // Attempt normalization via Value
                    let mut v: serde_json::Value =
                        serde_json::from_str(&structs_content).map_err(|e| {
                            TypeIntegrationError::ValidationFailed {
                                message: format!("Failed to parse structs.json: {}", e),
                            }
                        })?;

                    if let Some(structs_arr) = v.get_mut("structs").and_then(|s| s.as_array_mut()) {
                        for s in structs_arr.iter_mut() {
                            if let Some(fields_val) = s.get("fields").cloned() {
                                // Move fields -> members
                                if let Some(o) = s.as_object_mut() {
                                    o.insert("members".to_string(), fields_val);
                                    o.remove("fields");
                                }
                                if let Some(obj) = s.as_object_mut() {
                                    // Ensure struct has raw_content
                                    if !obj.contains_key("raw_content") {
                                        obj.insert(
                                            "raw_content".to_string(),
                                            serde_json::Value::String(String::new()),
                                        );
                                    }
                                    // Ensure struct has a category (required by StructDefinition)
                                    if !obj.contains_key("category") {
                                        obj.insert(
                                            "category".to_string(),
                                            serde_json::Value::String("struct".to_string()),
                                        );
                                    }
                                    // Ensure struct has is_alias (required by StructDefinition)
                                    if !obj.contains_key("is_alias") {
                                        obj.insert(
                                            "is_alias".to_string(),
                                            serde_json::Value::Bool(false),
                                        );
                                    }
                                }
                            }

                            // Normalize each member: field_type -> type_name
                            if let Some(members_arr) =
                                s.get_mut("members").and_then(|m| m.as_array_mut())
                            {
                                for member in members_arr.iter_mut() {
                                    if let Some(ft) = member.get("field_type").cloned() {
                                        if let Some(o) = member.as_object_mut() {
                                            o.insert("type_name".to_string(), ft);
                                            o.remove("field_type");
                                        }
                                        // Ensure required fields exist for StructMember
                                        if let Some(mobj) = member.as_object_mut() {
                                            if !mobj.contains_key("definition") {
                                                mobj.insert(
                                                    "definition".to_string(),
                                                    serde_json::Value::String(String::new()),
                                                );
                                            }
                                            if !mobj.contains_key("raw_content") {
                                                mobj.insert(
                                                    "raw_content".to_string(),
                                                    serde_json::Value::String(String::new()),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    serde_json::from_value(v).map_err(|e| {
                        TypeIntegrationError::ValidationFailed {
                            message: format!(
                                "Failed to parse (after normalization) structs.json: {}",
                                e
                            ),
                        }
                    })?
                }
            },
        };

    let enums_path = intermediate_dir.join("enums.json");
    log_info(&format!("Reading enums.json from {:?}", enums_path));
    let enums_content = std::fs::read_to_string(&enums_path)?;
    let enums_data: EnumData = match serde_json::from_str::<Vec<EnumDefinition>>(&enums_content) {
        Ok(vec) => EnumData { enums: vec },
        Err(_) => match serde_json::from_str::<EnumData>(&enums_content) {
            Ok(ed) => ed,
            Err(_) => {
                // Attempt normalization via Value to add missing raw_content/is_alias keys
                let mut v: serde_json::Value =
                    serde_json::from_str(&enums_content).map_err(|e| {
                        TypeIntegrationError::ValidationFailed {
                            message: format!("Failed to parse enums.json: {}", e),
                        }
                    })?;

                if let Some(enums_arr) = v.get_mut("enums").and_then(|e| e.as_array_mut()) {
                    for edef in enums_arr.iter_mut() {
                        if let Some(obj) = edef.as_object_mut() {
                            if !obj.contains_key("raw_content") {
                                obj.insert(
                                    "raw_content".to_string(),
                                    serde_json::Value::String(String::new()),
                                );
                            }
                            if !obj.contains_key("is_alias") {
                                obj.insert("is_alias".to_string(), serde_json::Value::Bool(false));
                            }
                            if !obj.contains_key("enum_type") {
                                obj.insert(
                                    "enum_type".to_string(),
                                    serde_json::Value::String("enum".to_string()),
                                );
                            }
                            // Normalize each value
                            if let Some(values_arr) =
                                edef.get_mut("values").and_then(|v| v.as_array_mut())
                            {
                                for val in values_arr.iter_mut() {
                                    if let Some(vobj) = val.as_object_mut() {
                                        if !vobj.contains_key("raw_content") {
                                            vobj.insert(
                                                "raw_content".to_string(),
                                                serde_json::Value::String(String::new()),
                                            );
                                        }
                                        if !vobj.contains_key("is_alias") {
                                            vobj.insert(
                                                "is_alias".to_string(),
                                                serde_json::Value::Bool(false),
                                            );
                                        }
                                        // Ensure value is a string (some tests use numbers)
                                        if let Some(value_field) = vobj.get("value") {
                                            if value_field.is_number() || value_field.is_boolean() {
                                                let s = value_field.to_string();
                                                vobj.insert(
                                                    "value".to_string(),
                                                    serde_json::Value::String(s),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                serde_json::from_value(v).map_err(|e| TypeIntegrationError::ValidationFailed {
                    message: format!("Failed to parse (after normalization) enums.json: {}", e),
                })?
            }
        },
    };

    let types_path = intermediate_dir.join("types.json");
    log_info(&format!("Reading types.json from {:?}", types_path));
    let types_content = std::fs::read_to_string(&types_path)?;
    let types_data: TypeData = match serde_json::from_str::<Vec<TypeDefinition>>(&types_content) {
        Ok(vec) => TypeData { types: vec },
        Err(_) => match serde_json::from_str::<TypeData>(&types_content) {
            Ok(td) => td,
            Err(_) => {
                // Attempt normalization via Value to add missing keys and map base_type -> definition
                let mut v: serde_json::Value =
                    serde_json::from_str(&types_content).map_err(|e| {
                        TypeIntegrationError::ValidationFailed {
                            message: format!("Failed to parse types.json: {}", e),
                        }
                    })?;

                if let Some(types_arr) = v.get_mut("types").and_then(|t| t.as_array_mut()) {
                    for tdef in types_arr.iter_mut() {
                        if let Some(obj) = tdef.as_object_mut() {
                            if !obj.contains_key("raw_content") {
                                obj.insert(
                                    "raw_content".to_string(),
                                    serde_json::Value::String(String::new()),
                                );
                            }
                            if !obj.contains_key("category") {
                                obj.insert(
                                    "category".to_string(),
                                    serde_json::Value::String("type".to_string()),
                                );
                            }
                            if !obj.contains_key("type_references") {
                                obj.insert(
                                    "type_references".to_string(),
                                    serde_json::Value::Array(vec![]),
                                );
                            }
                            if !obj.contains_key("is_alias") {
                                obj.insert("is_alias".to_string(), serde_json::Value::Bool(false));
                            }
                            // Map base_type -> definition if present
                            if let Some(base) = obj.get("base_type").cloned() {
                                if !obj.contains_key("definition") {
                                    obj.insert("definition".to_string(), base);
                                }
                            }
                        }
                    }
                }

                serde_json::from_value(v).map_err(|e| TypeIntegrationError::ValidationFailed {
                    message: format!("Failed to parse (after normalization) types.json: {}", e),
                })?
            }
        },
    };

    let constants_path = intermediate_dir.join("constants.json");
    log_info(&format!("Reading constants.json from {:?}", constants_path));
    let constants_content = std::fs::read_to_string(&constants_path)?;
    let _constants_data: serde_json::Value =
        serde_json::from_str(&constants_content).map_err(|e| {
            TypeIntegrationError::ValidationFailed {
                message: format!("Failed to parse constants.json: {}", e),
            }
        })?;

    let functions_path = intermediate_dir.join("functions.json");
    log_info(&format!("Reading functions.json from {:?}", functions_path));
    let functions_content = std::fs::read_to_string(&functions_path)?;
    let _functions_data: serde_json::Value =
        serde_json::from_str(&functions_content).map_err(|e| {
            TypeIntegrationError::ValidationFailed {
                message: format!("Failed to parse functions.json: {}", e),
            }
        })?;

    // Build a global type registry to track all defined and used types
    let mut defined_types: HashMap<String, &str> = HashMap::new();
    let mut used_types = HashSet::<String>::new();

    // 1. Collect all defined types
    // Structs
    for struct_def in &structs_data.structs {
        if let Some(existing_location) = defined_types.get(&struct_def.name) {
            return Err(TypeIntegrationError::DuplicateType {
                type_name: struct_def.name.clone(),
                locations: vec![existing_location.to_string(), "structs.json".to_string()],
            });
        }
        defined_types.insert(struct_def.name.clone(), "structs.json");

        // Collect types used in struct fields
        for field in &struct_def.members {
            if !is_primitive_type(&field.type_name) {
                used_types.insert(field.type_name.clone());
            }
        }
    }

    // Enums
    for enum_def in &enums_data.enums {
        if let Some(existing_location) = defined_types.get(&enum_def.name) {
            return Err(TypeIntegrationError::DuplicateType {
                type_name: enum_def.name.clone(),
                locations: vec![existing_location.to_string(), "enums.json".to_string()],
            });
        }
        defined_types.insert(enum_def.name.clone(), "enums.json");
    }

    // Types (typedefs/aliases)
    for type_def in &types_data.types {
        if let Some(existing_location) = defined_types.get(&type_def.name) {
            // Allow certain types to exist in multiple files (common in Vulkan spec)
            let is_allowed_duplicate = match (*existing_location, type_def.category.as_str()) {
                ("enums.json", "enum") => true, // Enum types can be in both enums.json and types.json
                ("structs.json", "struct") => true, // Struct types can be in both structs.json and types.json
                _ => false,
            };

            if is_allowed_duplicate {
                log_info(&format!(
                    "Allowing {} type '{}' to exist in both {} and types.json",
                    type_def.category, type_def.name, existing_location
                ));
                // Continue processing - this is allowed
            } else {
                return Err(TypeIntegrationError::DuplicateType {
                    type_name: type_def.name.clone(),
                    locations: vec![existing_location.to_string(), "types.json".to_string()],
                });
            }
        }
        defined_types.insert(type_def.name.clone(), "types.json");

        // Collect type references from definition if available
        if type_def.definition.is_some() {
            // Extract referenced types from the definition text
            // This is a simplified approach - the enhanced parsing should provide type_references
            for type_ref in &type_def.type_references {
                if !is_primitive_type(type_ref) {
                    used_types.insert(type_ref.clone());
                }
            }
        }
    }

    // 2. Check for undefined types that are referenced
    let mut undefined_types = Vec::new();
    for used_type in used_types {
        // Skip primitive types and opaque handles
        if !defined_types.contains_key(&used_type) && !is_opaque_handle(&used_type) {
            undefined_types.push(used_type.clone());
            log_debug(&format!("Type integration - Undefined type: {}", used_type));
        }
    }

    if !undefined_types.is_empty() {
        log_error(&format!(
            "Found {} undefined types referenced in the API",
            undefined_types.len()
        ));
        // In some cases, especially with Vulkan, we might have opaque handles or types
        // that are defined externally or have special handling. Log but don't fail.
        log_info("Continuing despite undefined types due to possible external definitions");
    }

    // 3. Check for circular dependencies in typedefs
    // (This would require a more complex algorithm to detect cycles in the type graph)

    // 4. Ensure enum values are valid
    for enum_def in &enums_data.enums {
        // Check for duplicate enum values
        let mut seen_values = HashSet::new();
        for enum_value in &enum_def.values {
            let value_str = enum_value.value.as_deref().unwrap_or("0");
            if !seen_values.insert(value_str.to_string()) {
                log_error(&format!(
                    "Duplicate enum value {} in enum {}",
                    value_str, enum_def.name
                ));
                // In Vulkan, some enums might have intentionally duplicated values for aliasing
                // Log but don't fail
            }
        }
    }

    log_info("Data consistency checks completed successfully");
    Ok(())
}

/// Perform comprehensive validation of a specific type across the codebase
/// This can be used to validate individual types when needed
pub fn validate_type(
    type_name: &str,
    intermediate_dir: &Path,
) -> TypeIntegrationResult<TypeValidationReport> {
    log_info(&format!(
        "Validating type '{}' across the codebase...",
        type_name
    ));

    let mut report = TypeValidationReport::new(type_name);

    // Check if the type is defined
    let defined_as = find_type_definition(type_name, intermediate_dir)?;
    report.definition_location = defined_as.clone();

    // Find all usages of the type
    let usages = find_type_usages(type_name, intermediate_dir)?;
    report.usages = usages;

    // Check for consistency in usage
    if !report.usages.is_empty() {
        log_debug(&format!(
            "Type Integration - Type '{}' is used in {} different places",
            type_name,
            report.usages.len()
        ));
    } else {
        log_debug(&format!(
            "Type Integration - Type '{}' is defined but not used anywhere",
            type_name
        ));
        report
            .warnings
            .push(format!("Type '{}' is defined but not used", type_name));
    }

    log_info(&format!("Validation of type '{}' completed", type_name));
    Ok(report)
}

/// Find where a type is defined
fn find_type_definition(type_name: &str, intermediate_dir: &Path) -> TypeIntegrationResult<String> {
    // Check in structs
    let structs_path = intermediate_dir.join("structs.json");
    let structs_data: StructData = serde_json::from_str(&std::fs::read_to_string(structs_path)?)?;

    for struct_def in &structs_data.structs {
        if struct_def.name == type_name {
            return Ok("structs.json".to_string());
        }
    }

    // Check in enums
    let enums_path = intermediate_dir.join("enums.json");
    let enums_data: EnumData = serde_json::from_str(&std::fs::read_to_string(enums_path)?)?;

    for enum_def in &enums_data.enums {
        if enum_def.name == type_name {
            return Ok("enums.json".to_string());
        }
    }

    // Check in types
    let types_path = intermediate_dir.join("types.json");
    let types_data: TypeData = serde_json::from_str(&std::fs::read_to_string(types_path)?)?;

    for type_def in &types_data.types {
        if type_def.name == type_name {
            return Ok("types.json".to_string());
        }
    }

    // If the type is primitive or an opaque handle, return a special marker
    if is_primitive_type(type_name) {
        return Ok("primitive".to_string());
    }

    if is_opaque_handle(type_name) {
        return Ok("opaque_handle".to_string());
    }

    Err(TypeIntegrationError::TypeNotFound {
        type_name: type_name.to_string(),
        referenced_in: "validation request".to_string(),
    })
}

/// Find all usages of a type
fn find_type_usages(
    type_name: &str,
    intermediate_dir: &Path,
) -> TypeIntegrationResult<Vec<TypeUsage>> {
    let mut usages = Vec::new();

    // Check in structs
    let structs_path = intermediate_dir.join("structs.json");
    let structs_data: StructData = serde_json::from_str(&std::fs::read_to_string(structs_path)?)?;

    for struct_def in &structs_data.structs {
        for field in &struct_def.members {
            if field.type_name == type_name {
                usages.push(TypeUsage {
                    location: format!("struct {} field {}", struct_def.name, field.name),
                    usage_type: "field_type".to_string(),
                    is_pointer: false, // Simplified - can't determine pointer level
                    is_const: false,   // Simplified - can't determine const qualification
                });
            }
        }
    }

    // Check in types (as base type)
    let types_path = intermediate_dir.join("types.json");
    let types_data: TypeData = serde_json::from_str(&std::fs::read_to_string(types_path)?)?;

    for type_def in &types_data.types {
        // Check if this type references the target type in its definition or type_references
        if type_def.type_references.contains(&type_name.to_string()) {
            usages.push(TypeUsage {
                location: format!("type alias {}", type_def.name),
                usage_type: "referenced_type".to_string(),
                is_pointer: false, // This information is now in the TypeDataKind, using default
                is_const: false,   // TypeDefinition doesn't have is_const, using default value
            });
        }
    }

    // Could add more checks for functions, etc.

    Ok(usages)
}

/// Report on type validation
#[derive(Debug)]
pub struct TypeValidationReport {
    pub type_name: String,
    pub definition_location: String,
    pub usages: Vec<TypeUsage>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl TypeValidationReport {
    pub fn new(type_name: &str) -> Self {
        Self {
            type_name: type_name.to_string(),
            definition_location: String::new(),
            usages: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Structure representing a usage of a type
#[derive(Debug)]
pub struct TypeUsage {
    pub location: String,
    pub usage_type: String,
    pub is_pointer: bool,
    pub is_const: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_is_primitive_type() {
        assert!(is_primitive_type("void"));
        assert!(is_primitive_type("uint32_t"));
        assert!(is_primitive_type("float"));
        assert!(!is_primitive_type("VkInstance"));
        assert!(!is_primitive_type("SomeCustomType"));
    }

    #[test]
    fn test_is_opaque_handle() {
        assert!(is_opaque_handle("VkInstance"));
        assert!(is_opaque_handle("VkDevice"));
        assert!(is_opaque_handle("VkSomeHandle"));
        assert!(!is_opaque_handle("uint32_t"));
        assert!(!is_opaque_handle("VkStructType")); // Not a handle
    }

    #[test]
    fn test_verify_intermediate_files() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary directory
        let temp_dir = tempdir()?;
        let dir_path = temp_dir.path();

        // Should fail when files don't exist
        let result = verify_intermediate_files(dir_path);
        assert!(result.is_err());

        // Create dummy files
        let required_files = [
            "structs.json",
            "enums.json",
            "constants.json",
            "functions.json",
            "types.json",
            "extensions.json",
            "macros.json",
            "includes.json",
        ];

        for file in &required_files {
            let file_path = dir_path.join(file);
            let mut file = fs::File::create(file_path)?;
            write!(file, "{{}}")?; // Empty JSON object
        }

        // Should succeed when all files exist
        let result = verify_intermediate_files(dir_path);
        assert!(result.is_ok());

        Ok(())
    }

    #[test]
    fn test_data_consistency() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary directory
        let temp_dir = tempdir()?;
        let dir_path = temp_dir.path();

        // Create test files with valid content
        // structs.json
        let structs_json = r#"{
            "structs": [
                {
                    "name": "VkTestStruct",
                    "fields": [
                        {
                            "name": "testField",
                            "field_type": "uint32_t",
                            "is_const": false,
                            "is_pointer": false,
                            "source_line": 10
                        },
                        {
                            "name": "testEnumField",
                            "field_type": "VkTestEnum",
                            "is_const": false,
                            "is_pointer": false,
                            "source_line": 11
                        }
                    ],
                    "source_line": 9
                }
            ]
        }"#;

        // enums.json
        let enums_json = r#"{
            "enums": [
                {
                    "name": "VkTestEnum",
                    "values": [
                        {
                            "name": "VK_TEST_VALUE_A",
                            "value": 0,
                            "source_line": 20
                        },
                        {
                            "name": "VK_TEST_VALUE_B",
                            "value": 1,
                            "source_line": 21
                        }
                    ],
                    "source_line": 19
                }
            ]
        }"#;

        // types.json
        let types_json = r#"{
            "types": [
                {
                    "name": "VkTestType",
                    "base_type": "uint32_t",
                    "is_const": false,
                    "is_pointer": false,
                    "source_line": 30
                }
            ]
        }"#;

        // Create empty files for the rest
        let empty_json = "{}";

        // Write the files
        fs::write(dir_path.join("structs.json"), structs_json)?;
        fs::write(dir_path.join("enums.json"), enums_json)?;
        fs::write(dir_path.join("types.json"), types_json)?;
        fs::write(dir_path.join("constants.json"), empty_json)?;
        fs::write(dir_path.join("functions.json"), empty_json)?;
        fs::write(dir_path.join("extensions.json"), empty_json)?;
        fs::write(dir_path.join("macros.json"), empty_json)?;
        fs::write(dir_path.join("includes.json"), empty_json)?;

        // Test data consistency check
        let result = check_data_consistency(dir_path);
        if let Err(e) = &result {
            println!("DEBUG: check_data_consistency returned error: {:?}", e);
        }
        assert!(result.is_ok());

        // Now create a duplicate type to test the error case
        let bad_types_json = r#"{
            "types": [
                {
                    "name": "VkTestEnum",
                    "base_type": "uint32_t",
                    "is_const": false,
                    "is_pointer": false,
                    "source_line": 40
                }
            ]
        }"#;

        fs::write(dir_path.join("types.json"), bad_types_json)?;

        // This should fail with a duplicate type error
        let result = check_data_consistency(dir_path);
        assert!(result.is_err());
        match result {
            Err(TypeIntegrationError::DuplicateType { type_name, .. }) => {
                assert_eq!(type_name, "VkTestEnum");
            }
            _ => panic!("Expected DuplicateType error"),
        }

        Ok(())
    }
}
