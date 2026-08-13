//! Struct generator module
//!
//! Generates Rust struct definitions from structs.json intermediate file

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorModule, GeneratorResult};

use crate::parser::vk_types::{EnumDefinition, StructDefinition, TypeDefinition};

/// Map from enum name -> the variant identifier to use for `Default::default()`.
/// For enums where 0 is not a valid value, this lets struct Default impls
/// avoid undefined behavior from `mem::zeroed()`.
type EnumDefaultMap = std::collections::HashMap<String, String>;

/// Set of `VkStructureType` variant C names that are actually emitted
/// into the generated `VkStructureType` enum. Used to skip
/// `PNextChainable` impls for structs whose sType belongs to a disabled
/// API variant (e.g. Vulkan SC) that vulkane doesn't compile in.
type KnownStructureTypes = std::collections::HashSet<String>;

/// Sanitize a type name to be a valid Rust identifier
fn sanitize_type_name(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_alphanumeric() || c == '_' {
            s.push(c);
        } else {
            s.push('_');
        }
    }
    // Prevent leading digits
    if s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        s = format!("_{}", s);
    }
    s
}

/// Generator module for Vulkan structs
pub struct StructGenerator;

impl Default for StructGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl StructGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Rust code for a single struct
    fn generate_struct(
        &self,
        struct_def: &StructDefinition,
        _all_type_names: &std::collections::HashSet<String>,
        enum_defaults: &EnumDefaultMap,
        known_structure_types: &KnownStructureTypes,
        enum_type_names: &std::collections::HashSet<String>,
        written_by_command: &std::collections::HashSet<String>,
        _output_dir: &Path,
    ) -> String {
        let mut code = String::new();

        let _is_union = struct_def.category == "union";

        // Determine which traits to derive.
        // We omit Debug because some structs contain union fields which
        // don't implement Debug, and we can't know at generation time
        // which fields are unions vs structs (they're just type names).
        let can_derive_copy = self.can_derive_copy(struct_def);
        let derives = if can_derive_copy {
            "#[derive(Clone, Copy)]"
        } else {
            "#[derive(Clone)]"
        };

        let is_union = struct_def.category == "union";
        let keyword = if is_union { "union" } else { "struct" };

        // Emit doc comment from vk.xml if present
        if let Some(comment) = &struct_def.comment {
            for line in comment.lines() {
                code.push_str(&format!(
                    "/// {}\n",
                    crate::codegen::sanitize_doc_line(line)
                ));
            }
        }

        code.push_str("#[repr(C)]\n");
        code.push_str(&format!("{}\n", derives));
        let sanitized_struct_name = sanitize_type_name(&struct_def.name);
        code.push_str(&format!("pub {} {} {{\n", keyword, sanitized_struct_name));

        // Fields with deduplication
        let mut seen_fields = std::collections::HashSet::new();

        for field in &struct_def.members {
            let field_name = self.escape_rust_keyword(&field.name);

            // Skip duplicate fields
            if seen_fields.contains(&field_name) {
                continue;
            }
            seen_fields.insert(field_name.clone());

            // Emit doc comment from vk.xml if present
            if let Some(comment) = &field.comment {
                for line in comment.lines() {
                    code.push_str(&format!(
                        "    /// {}\n",
                        crate::codegen::sanitize_doc_line(line)
                    ));
                }
            }

            let rust_type = self.map_member_type(field);

            // A field the driver fills cannot be typed as a Rust enum: an
            // implementation newer than this vk.xml may report a value outside
            // the declared set, and reading that is UB. See
            // `is_driver_written_enum_field`.
            if self.is_driver_written_enum_field(
                struct_def,
                field,
                &rust_type,
                enum_type_names,
                written_by_command,
            ) {
                code.push_str(&format!(
                    "    /// Raw `{}` value, written by the implementation.\n",
                    rust_type
                ));
                code.push_str(&format!(
                    "    ///\n    /// Typed as `i32` rather than `{}` because a driver may report a\n\
                     \x20   /// value this spec revision does not define, and reading an out-of-range\n\
                     \x20   /// discriminant as a Rust enum is undefined behaviour. Convert with\n\
                     \x20   /// [`{}::from_raw`] and handle `None`.\n",
                    rust_type, rust_type
                ));
                code.push_str(&format!("    pub {}: i32,\n", field_name));
                continue;
            }

            code.push_str(&format!("    pub {}: {},\n", field_name, rust_type));
        }

        code.push_str("}\n\n");

        // Default implementation
        code.push_str(&format!("impl Default for {} {{\n", sanitized_struct_name));
        code.push_str("    fn default() -> Self {\n");

        if is_union {
            // Unions must use zeroed() since only one field can be initialized
            code.push_str("        unsafe { std::mem::zeroed() }\n");
        } else {
            code.push_str("        Self {\n");

            let mut seen_fields = std::collections::HashSet::new();
            for field in &struct_def.members {
                let field_name = self.escape_rust_keyword(&field.name);

                if seen_fields.contains(&field_name) {
                    continue;
                }
                seen_fields.insert(field_name.clone());

                let rust_type = self.map_member_type(field);
                let is_pointer = rust_type.starts_with("*const") || rust_type.starts_with("*mut");
                let is_array = rust_type.starts_with('[');
                // Check if this field's base type is an enum we know how to default.
                // Only applies to scalar fields — array fields keep using zeroed().
                let default_value = if self.is_driver_written_enum_field(
                    struct_def,
                    field,
                    &rust_type,
                    enum_type_names,
                    written_by_command,
                ) {
                    // Now an `i32`, so the "0 is not a valid variant" problem
                    // that `enum_defaults` exists to dodge cannot arise: no
                    // integer is an invalid `i32`. Zero it and let the driver
                    // overwrite it, which is what the caller does anyway.
                    "0".to_string()
                } else if !is_pointer && !is_array && enum_defaults.contains_key(&field.type_name) {
                    let variant = &enum_defaults[&field.type_name];
                    format!("{}::{}", field.type_name, variant)
                } else {
                    self.get_default_value_for_rust_type(&rust_type, is_pointer)
                };
                code.push_str(&format!("            {}: {},\n", field_name, default_value));
            }

            code.push_str("        }\n");
        }

        code.push_str("    }\n");
        code.push_str("}\n\n");

        // Emit `unsafe impl PNextChainable` for every struct whose first
        // two fields are `sType: VkStructureType` and `pNext: *mut c_void`
        // AND whose sType has a fixed `values="VK_STRUCTURE_TYPE_..."`
        // attribute in vk.xml. This captures every extension / feature
        // struct that can participate in a pNext chain, and skips:
        //   - the generic base structs `VkBaseOutStructure` / `VkBaseInStructure`
        //     which don't carry a fixed sType,
        //   - any struct without the chain header layout,
        //   - structs whose sType lives in a disabled API variant
        //     (e.g. Vulkan SC), so the emitted impl never references a
        //     variant that isn't in the generated `VkStructureType`.
        if let Some(impl_code) =
            self.try_emit_pnext_chainable_impl(struct_def, known_structure_types)
        {
            code.push_str(&impl_code);
        }

        code
    }

    /// If `struct_def` is layout-compatible with the Vulkan `pNext`
    /// chain header (sType+pNext first) and its sType has a fixed
    /// `values="VK_STRUCTURE_TYPE_..."` attribute, produce the
    /// `unsafe impl PNextChainable` block. Returns `None` if the struct
    /// doesn't qualify.
    ///
    /// The emitted impl lives in the same file as the struct definition,
    /// which is `include!`'d into `crate::raw::bindings`, so the
    /// `super::pnext::PNextChainable` path resolves to
    /// `crate::raw::pnext::PNextChainable` at compile time.
    fn try_emit_pnext_chainable_impl(
        &self,
        struct_def: &StructDefinition,
        known_structure_types: &KnownStructureTypes,
    ) -> Option<String> {
        // Skip aliases and unions — only concrete structs can carry the
        // chain header.
        if struct_def.is_alias || struct_def.category == "union" {
            return None;
        }

        // Need at least two members to even consider this.
        if struct_def.members.len() < 2 {
            return None;
        }

        let first = &struct_def.members[0];
        let second = &struct_def.members[1];

        // First field must be `sType: VkStructureType`.
        if first.name != "sType" || first.type_name != "VkStructureType" {
            return None;
        }

        // Second field must be `pNext`. We don't insist on an exact type
        // spelling because vk.xml encodes it as `void*` with optional
        // const qualification; the struct itself is emitted with either
        // `*mut c_void` or `*const c_void` depending on direction, but
        // both layouts are identical for pNext-chain purposes.
        if second.name != "pNext" {
            return None;
        }

        // First field must have a fixed sType value (e.g.
        // `values="VK_STRUCTURE_TYPE_APPLICATION_INFO"`). Skip any
        // generic base structs (VkBaseInStructure / VkBaseOutStructure)
        // that omit this attribute.
        let stype_c_name_raw = first.values.as_deref()?;
        // Spec values are comma-separated in principle; we only use the
        // first entry (no struct in the current spec carries multiple).
        let stype_c_name = stype_c_name_raw.split(',').next()?.trim();
        if stype_c_name.is_empty() {
            return None;
        }
        // Skip structs whose sType belongs to a disabled API variant —
        // otherwise the impl would reference a variant that doesn't exist
        // in the emitted `VkStructureType` enum and fail to compile.
        if !known_structure_types.contains(stype_c_name) {
            return None;
        }
        let variant = stype_c_name
            .strip_prefix("VK_")
            .unwrap_or(stype_c_name)
            .to_string();

        let struct_name = sanitize_type_name(&struct_def.name);
        Some(format!(
            "unsafe impl super::pnext::PNextChainable for {struct_name} {{\n\
             \x20   const STRUCTURE_TYPE: VkStructureType = VkStructureType::{variant};\n\
             }}\n\n"
        ))
    }

    /// Determine if a struct can derive Copy trait
    fn can_derive_copy(&self, struct_def: &StructDefinition) -> bool {
        // Check all fields to see if they can be copied
        for field in &struct_def.members {
            let rust_type = self.simple_map_type(&field.type_name);

            // For simplified version, assume most Vulkan types can be copied
            if !self.type_supports_copy_simple(&rust_type) {
                return false;
            }
        }
        true
    }

    /// Check if a simple type supports Copy trait
    fn type_supports_copy_simple(&self, type_name: &str) -> bool {
        match type_name {
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64"
            | "bool" | "usize" | "isize" | "c_char" | "c_uchar" | "c_short" | "c_ushort"
            | "c_int" | "c_uint" | "c_long" | "c_ulong" | "c_longlong" | "c_ulonglong"
            | "c_float" | "c_double" | "c_void" => true,

            // Pointers support Copy
            _ if type_name.starts_with("*const") || type_name.starts_with("*mut") => true,

            // Most Vulkan types should support Copy (they're typically enums or simple handles)
            _ if type_name.starts_with("Vk") => true,

            // Be conservative for unknown types
            _ => false,
        }
    }

    /// Map Vulkan types to Rust types with proper array handling.
    ///
    /// `array_dims` is the list of array dimensions in C-declaration
    /// order (outermost first). For a C declaration `float matrix[3][4]`
    /// the dims are `["3", "4"]` and the resulting Rust type is
    /// `[[f32; 4]; 3]` — preserving C's row-major layout.
    fn map_type_to_rust(
        &self,
        vulkan_type: &str,
        const_qualified: bool,
        pointer_level: usize,
        array_dims: &[String],
    ) -> String {
        // Handle arrays first — nest innermost-to-outermost so the
        // outermost C dimension ends up as the outermost Rust array.
        if !array_dims.is_empty() && pointer_level == 0 {
            let base_type = self.map_base_vulkan_to_rust(vulkan_type);
            let mut result = base_type;
            for dim in array_dims.iter().rev() {
                result = format!("[{}; {}]", result, dim);
            }
            return result;
        }

        // Handle pointers
        let base_type = self.map_base_vulkan_to_rust(vulkan_type);

        if pointer_level == 0 {
            base_type
        } else {
            let mut result = base_type;
            for level in 0..pointer_level {
                // Apply const qualification to the outer-most pointer when requested.
                // Build from inner to outer; outermost iteration is when level == pointer_level - 1.
                if level == pointer_level - 1 {
                    if const_qualified {
                        result = format!("*const {}", result);
                    } else {
                        result = format!("*mut {}", result);
                    }
                } else {
                    result = format!("*mut {}", result);
                }
            }
            result
        }
    }

    /// Map base Vulkan types to Rust types
    fn map_base_vulkan_to_rust(&self, vulkan_type: &str) -> String {
        match vulkan_type {
            // Use fully-qualified names to avoid relying on a specific import
            // order in the final assembled file.
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
            _ => vulkan_type.to_string(), // Keep Vulkan types as-is
        }
    }

    /// Simple type mapping for simplified intermediate types
    fn simple_map_type(&self, type_name: &str) -> String {
        self.map_base_vulkan_to_rust(type_name)
    }

    /// Parse a struct member definition to produce the full Rust type.
    /// Handles pointers, const, and arrays from the C definition string.
    ///
    /// Multi-dimensional arrays are supported: `float matrix[3][4]` →
    /// `[[f32; 4]; 3]` (preserving C row-major layout).
    fn map_member_type(&self, member: &crate::parser::vk_types::StructMember) -> String {
        let def = member.definition.trim();
        let _base = self.map_base_vulkan_to_rust(&member.type_name);

        // Count pointer levels
        let pointer_level = def.chars().filter(|c| *c == '*').count();

        // Check const qualification
        let const_qualified = def.starts_with("const") || def.contains("const ");

        // Collect every `[N]` group from the declaration in order.
        // C allows `T name[a][b]` which means "array of `a` elements,
        // each an array of `b`"; we keep the dimensions in declaration
        // order and let `map_type_to_rust` nest them correctly.
        let mut array_dims: Vec<String> = Vec::new();
        let mut rest = def;
        while let Some(start) = rest.find('[') {
            if let Some(end) = rest[start..].find(']') {
                let size_str = rest[start + 1..start + end].trim();
                let dim =
                    if !size_str.is_empty() && !size_str.chars().next().unwrap().is_ascii_digit() {
                        format!("{} as usize", size_str)
                    } else {
                        size_str.to_string()
                    };
                if !dim.is_empty() {
                    array_dims.push(dim);
                }
                rest = &rest[start + end + 1..];
            } else {
                break;
            }
        }

        self.map_type_to_rust(
            &member.type_name,
            const_qualified,
            pointer_level,
            &array_dims,
        )
    }

    /// Get default value for a fully mapped Rust type (including arrays)
    fn get_default_value_for_rust_type(&self, rust_type: &str, is_pointer: bool) -> String {
        if is_pointer {
            // Use null_mut for mutable pointers, null for const pointers
            if rust_type.starts_with("*const") {
                return "std::ptr::null()".to_string();
            } else {
                return "std::ptr::null_mut()".to_string();
            }
        }

        // Handle array types like [VkMemoryType; 32]
        if rust_type.starts_with('[') && rust_type.contains(';') && rust_type.ends_with(']') {
            // Extract the array type and size
            if let Some(semicolon_pos) = rust_type.find(';') {
                let _inner_type = rust_type[1..semicolon_pos].trim();
                let _size_part = rust_type[semicolon_pos + 1..rust_type.len() - 1].trim();

                // Use unsafe zeroed for all array defaults since array elements
                // may be complex types (structs, enums) that don't implement Default
                return "unsafe { std::mem::zeroed() }".to_string();
            }
        }

        // Handle standard types
        match rust_type {
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => "0".to_string(),
            "f32" | "f64" => "0.0".to_string(),
            "bool" => "false".to_string(),
            "c_char" | "c_uchar" | "c_short" | "c_ushort" | "c_int" | "c_uint" | "c_long"
            | "c_ulong" | "c_longlong" | "c_ulonglong" => "0".to_string(),
            "c_float" => "0.0".to_string(),
            "c_double" => "0.0".to_string(),
            _ => "unsafe { std::mem::zeroed() }".to_string(),
        }
    }

    /// Escape Rust keywords by adding r# prefix
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

    /// Names of every type emitted as a *Rust `enum`* (not a bitmask, which
    /// becomes an integer newtype plus `pub const`s).
    ///
    /// This set is what makes [`Self::is_driver_written_enum_field`] possible.
    /// A Rust `enum` has a closed set of valid bit patterns; a bitmask does
    /// not, and neither do handles or nested structs, so only these types are
    /// unsound to have a driver write into.
    fn build_enum_type_names(&self, input_dir: &Path) -> std::collections::HashSet<String> {
        let mut names = std::collections::HashSet::new();
        let enums_path = input_dir.join("enums.json");
        let Ok(content) = fs::read_to_string(&enums_path) else {
            return names;
        };
        let Ok(enums) = serde_json::from_str::<Vec<EnumDefinition>>(&content) else {
            return names;
        };
        for e in &enums {
            if e.enum_type == "bitmask" {
                continue;
            }
            names.insert(e.name.clone());
        }
        names
    }

    /// Whether this field must be emitted as a raw `i32` rather than as its
    /// Rust `enum` type, because the *driver* writes it.
    ///
    /// Reading a Rust `enum` whose memory holds a discriminant outside its
    /// declared set is undefined behaviour. For a struct that `vk.xml` marks
    /// `returnedonly="true"`, the implementation fills the field, and it is
    /// free to report a value this `vk.xml` has never heard of — a component
    /// type or driver ID from an extension newer than the pinned spec. That
    /// makes the UB reachable by *upgrading a graphics driver*, with no
    /// application change and no error path to observe.
    ///
    /// Emitting `i32` moves the decision to the caller, who must go through
    /// the generated `from_raw` and handle `None`. The check already existed;
    /// typing the field as the enum guaranteed it could never run.
    ///
    /// Two deliberate exclusions:
    ///
    /// - **`sType`** — the application writes it before the call even in a
    ///   `returnedonly` struct (that is how a `pNext` query chain is built),
    ///   so it is never driver-authored and keeps its ergonomic enum type.
    /// - **pointers and arrays** — the pointee is driver-written too, but
    ///   reading through a raw pointer is already `unsafe` and carries its own
    ///   contract. Narrowing to direct value fields keeps this change to the
    ///   surface that is unsound to touch from *safe* code.
    fn is_driver_written_enum_field(
        &self,
        struct_def: &StructDefinition,
        field: &crate::parser::vk_types::StructMember,
        rust_type: &str,
        enum_type_names: &std::collections::HashSet<String>,
        written_by_command: &std::collections::HashSet<String>,
    ) -> bool {
        let marked = struct_def.returnedonly.as_deref() == Some("true");
        if !marked && !written_by_command.contains(&struct_def.name) {
            return false;
        }
        if field.name == "sType" {
            return false;
        }
        // Only a bare enum-typed field; `map_member_type` has already folded
        // pointers and arrays into the spelling, so anything decorated is out.
        enum_type_names.contains(rust_type)
    }

    /// Structs the implementation writes into, **derived from command
    /// signatures** rather than from the registry's own marker.
    ///
    /// `returnedonly="true"` is authoritative when present, and it was the sole
    /// rule here originally. It is not always present. `VkCooperativeVectorPropertiesNV`
    /// is filled by `vkGetPhysicalDeviceCooperativeVectorPropertiesNV` and
    /// carries no marker, while its sibling `VkPhysicalDeviceCooperativeVectorPropertiesNV`
    /// does — so keying only on the marker left five `VkComponentTypeKHR`
    /// fields typed as a Rust enum that a driver fills. A driver reporting a
    /// component type this `vk.xml` does not define makes reading them
    /// undefined behaviour, which is the exact defect the marker-based rule was
    /// written to remove.
    ///
    /// So the marker is now a *source* of the answer rather than the whole of
    /// it: a struct also counts as driver-written when it appears as a
    /// **non-const pointer parameter** of a `vkGet*` or `vkEnumerate*` command,
    /// which is what "the implementation fills this for you" looks like in the
    /// C signature. Derived, so a future `vk.xml` that adds another
    /// under-annotated query struct is covered without anyone noticing and
    /// hand-listing it.
    ///
    /// Measured against `vk.xml` at header 348 the derived set adds exactly two
    /// structs beyond the marked ones — `VkCooperativeVectorPropertiesNV` (5
    /// fields) and `VkDataGraphPipelinePropertyQueryResultARM` (1) — so this is
    /// a surgical correction, not a reclassification of the binding surface.
    fn build_driver_written_structs(&self, input_dir: &Path) -> std::collections::HashSet<String> {
        use crate::parser::vk_types::VulkanCommand;

        let mut written = std::collections::HashSet::new();
        let path = input_dir.join("functions.json");
        // Deliberately fatal rather than a soft fallback. A silent empty set
        // here degrades exactly one thing — whether driver-written enum fields
        // are typed `i32` — and it degrades it *invisibly*: the bindings still
        // compile, the tests still pass, and the UB comes back. The whole point
        // of this function is to remove a defect that produced no symptom, so
        // it must not fail in a way that produces no symptom.
        let content = fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read {} ({e}) — the driver-written struct set would be \
                 empty and enum fields a driver fills would silently regain \
                 their UB-prone typing",
                path.display()
            )
        });
        let commands: Vec<VulkanCommand> = serde_json::from_str(&content).unwrap_or_else(|e| {
            panic!(
                "cannot parse {} as commands ({e}) — see above; an empty set \
                 here is silently unsound, not merely incomplete",
                path.display()
            )
        });

        for command in &commands {
            if command.is_alias {
                continue;
            }
            if !(command.name.starts_with("vkGet") || command.name.starts_with("vkEnumerate")) {
                continue;
            }
            for param in &command.parameters {
                // `const T*` is an input the caller fills; `T*` is an output the
                // implementation fills. That distinction is the whole signal.
                let decl = &param.definition;
                if decl.contains('*') && !decl.contains("const") {
                    written.insert(param.type_name.clone());
                }
            }
        }
        written
    }

    /// Build a map from enum name to the variant identifier to use for `Default::default()`.
    /// For Rust-enum-style enums (not bitmask), we record the first non-alias variant
    /// only when 0 is not a valid value — meaning `mem::zeroed()` would produce UB.
    fn build_enum_defaults(&self, input_dir: &Path) -> EnumDefaultMap {
        let mut map = EnumDefaultMap::new();
        let enums_path = input_dir.join("enums.json");
        let content = match fs::read_to_string(&enums_path) {
            Ok(c) => c,
            Err(_) => return map,
        };
        let enums: Vec<EnumDefinition> = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return map,
        };

        for e in &enums {
            // Only handle Rust-enum-style enums (not bitmask, which are emitted as `pub const`).
            // Bitmask enums get u32/u64 default to 0 which is always a valid empty flag set.
            if e.enum_type == "bitmask" {
                continue;
            }

            // Skip if any variant has value 0 — `mem::zeroed()` is safe in that case.
            let mut has_zero = false;
            let mut first_non_alias: Option<&str> = None;
            for v in &e.values {
                if v.is_alias {
                    continue;
                }
                if first_non_alias.is_none() {
                    first_non_alias = Some(&v.name);
                }
                if let Some(val_str) = &v.value {
                    if val_str.trim() == "0" {
                        has_zero = true;
                        break;
                    }
                }
            }
            if has_zero {
                continue;
            }
            if let Some(first_name) = first_non_alias {
                // Format the variant name the same way the enum generator does:
                // strip leading "VK_" prefix.
                let variant = first_name
                    .strip_prefix("VK_")
                    .unwrap_or(first_name)
                    .to_string();
                map.insert(e.name.clone(), variant);
            }
        }
        map
    }

    /// Build the set of `VkStructureType` variant C names present in
    /// the parsed enums. Used to skip `PNextChainable` impls whose sType
    /// was stripped by the api-variant filter in the parser.
    fn build_known_structure_types(&self, input_dir: &Path) -> KnownStructureTypes {
        let mut set = KnownStructureTypes::new();
        let enums_path = input_dir.join("enums.json");
        let Ok(content) = fs::read_to_string(&enums_path) else {
            return set;
        };
        let Ok(enums) = serde_json::from_str::<Vec<EnumDefinition>>(&content) else {
            return set;
        };
        for e in &enums {
            if e.name != "VkStructureType" {
                continue;
            }
            for v in &e.values {
                set.insert(v.name.clone());
            }
        }
        set
    }

    /// Generate code for all structs in the input directory
    fn generate_all_structs(
        &self,
        input_dir: &Path,
        output_dir: &Path,
        all_type_names: &std::collections::HashSet<String>,
    ) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("structs.json");
        let input_content = fs::read_to_string(&input_path).map_err(GeneratorError::Io)?;

        // Parse JSON - try direct array format first, then fallback to object-with-array { "structs": [...] }
        let structs: Vec<StructDefinition> =
            match serde_json::from_str::<Vec<StructDefinition>>(&input_content) {
                Ok(v) => v,
                Err(_) => {
                    #[derive(serde::Deserialize)]
                    struct StructsFile {
                        structs: Vec<StructDefinition>,
                    }

                    let wrapper: StructsFile =
                        serde_json::from_str(&input_content).map_err(GeneratorError::Json)?;
                    wrapper.structs
                }
            };

        // Build the enum-defaults map by reading enums.json. For enums where 0
        // is not a valid variant, we record the first variant name so struct
        // Default impls can use it instead of `mem::zeroed()` (which is UB).
        let enum_defaults = self.build_enum_defaults(input_dir);

        // Build the set of VkStructureType variants that actually made it
        // into the emitted enum. The parser already drops vulkansc-only
        // `<feature>` blocks, so variants they introduced aren't present;
        // we use this set to suppress PNextChainable impls that would
        // otherwise reference missing variants.
        let known_structure_types = self.build_known_structure_types(input_dir);

        // Which types are Rust enums, so driver-written fields of those types
        // can be emitted as raw integers instead. See
        // `is_driver_written_enum_field`.
        let enum_type_names = self.build_enum_type_names(input_dir);
        let written_by_command = self.build_driver_written_structs(input_dir);

        // Generate code
        let mut generated_code = String::new();

        // Don't add imports here - they're handled by the assembler

        // Add allow directives (outer attributes)
        generated_code.push_str("#[allow(non_camel_case_types)]\n");
        generated_code.push_str("#[allow(dead_code)]\n");

        // Generate structs
        for struct_def in &structs {
            generated_code.push_str(&self.generate_struct(
                struct_def,
                all_type_names,
                &enum_defaults,
                &known_structure_types,
                &enum_type_names,
                &written_by_command,
                output_dir,
            ));
        }

        // Ensure output directory exists
        fs::create_dir_all(output_dir).map_err(GeneratorError::Io)?;

        // Write output file
        let output_path = output_dir.join("structs.rs");
        fs::write(output_path, generated_code).map_err(GeneratorError::Io)?;

        crate::codegen::logging::log_info(&format!(
            "StructGeneratorModule: Generated {} structs",
            structs.len()
        ));
        Ok(())
    }
}

impl GeneratorModule for StructGenerator {
    fn name(&self) -> &str {
        "StructGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["structs.json".to_string()]
    }

    fn output_file(&self) -> String {
        "structs.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        vec![
            "TypeGenerator".to_string(),
            "EnumGenerator".to_string(),
            "ConstantGenerator".to_string(),
        ]
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Collect all type names from types.json and structs.json for reference
        let mut all_type_names = std::collections::HashSet::new();
        // Read types.json
        let types_path = input_dir.join("types.json");
        if let Ok(types_content) = fs::read_to_string(types_path) {
            if let Ok(types) = serde_json::from_str::<Vec<TypeDefinition>>(&types_content) {
                for t in &types {
                    all_type_names.insert(t.name.clone());
                }
            }
        }
        // Read structs.json
        let structs_path = input_dir.join("structs.json");
        if let Ok(structs_content) = fs::read_to_string(structs_path) {
            if let Ok(structs) = serde_json::from_str::<Vec<StructDefinition>>(&structs_content) {
                for s in &structs {
                    all_type_names.insert(s.name.clone());
                }
            }
        }
        self.generate_all_structs(input_dir, output_dir, &all_type_names)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_mapping() {
        let generator = StructGenerator::new();

        assert_eq!(generator.map_type_to_rust("uint32_t", false, 0, &[]), "u32");
        assert_eq!(
            generator.map_type_to_rust("uint32_t", true, 1, &[]),
            "*const u32"
        );
        assert_eq!(
            generator.map_type_to_rust("uint32_t", false, 1, &[]),
            "*mut u32"
        );
        assert_eq!(
            generator.map_type_to_rust("void", true, 1, &[]),
            "*const c_void"
        );
        assert_eq!(
            generator.map_type_to_rust("VkDevice", false, 0, &[]),
            "VkDevice"
        );

        // Test multiple pointer levels
        assert_eq!(
            generator.map_type_to_rust("char", true, 2, &[]),
            "*const *mut c_char"
        );
        assert_eq!(
            generator.map_type_to_rust("void", false, 3, &[]),
            "*mut *mut *mut c_void"
        );

        // 1-D array
        assert_eq!(
            generator.map_type_to_rust("char", false, 0, &["256".to_string()]),
            "[c_char; 256]"
        );
        // 2-D array — matches VkTransformMatrixKHR's `float matrix[3][4]`.
        // Outermost C dimension = 3 stays outer in Rust, giving rows of 4.
        assert_eq!(
            generator.map_type_to_rust("float", false, 0, &["3".to_string(), "4".to_string()]),
            "[[f32; 4]; 3]"
        );
        // 3-D array — rare but must nest consistently.
        assert_eq!(
            generator.map_type_to_rust(
                "uint32_t",
                false,
                0,
                &["2".to_string(), "3".to_string(), "4".to_string()],
            ),
            "[[[u32; 4]; 3]; 2]"
        );
    }

    /// The derived driver-written set must find a struct that `vk.xml` fails to
    /// mark `returnedonly`. `VkCooperativeVectorPropertiesNV` is the real case:
    /// filled by `vkGetPhysicalDeviceCooperativeVectorPropertiesNV`, unmarked,
    /// and five `VkComponentTypeKHR` fields wide.
    #[test]
    fn driver_written_set_is_derived_from_output_pointer_params() {
        let dir = tempfile::TempDir::new().unwrap();
        let json = r#"[
          {
            "name": "vkGetPhysicalDeviceCooperativeVectorPropertiesNV",
            "return_type": "VkResult",
            "comment": null, "successcodes": null, "errorcodes": null,
            "alias": null, "api": null, "deprecated": null,
            "cmdbufferlevel": null, "pipeline": null, "queues": null,
            "renderpass": null, "videocoding": null,
            "raw_content": "", "is_alias": false,
            "parameters": [
              {"name":"physicalDevice","type_name":"VkPhysicalDevice","optional":null,"len":null,
               "altlen":null,"externsync":null,"noautovalidity":null,"objecttype":null,"stride":null,
               "validstructs":null,"api":null,"deprecated":null,"comment":null,
               "definition":"VkPhysicalDevice physicalDevice","raw_content":""},
              {"name":"pProperties","type_name":"VkCooperativeVectorPropertiesNV","optional":"true",
               "len":"pPropertyCount","altlen":null,"externsync":null,"noautovalidity":null,
               "objecttype":null,"stride":null,"validstructs":null,"api":null,"deprecated":null,
               "comment":null,
               "definition":"VkCooperativeVectorPropertiesNV * pProperties","raw_content":""}
            ]
          },
          {
            "name": "vkCreateInstance",
            "return_type": "VkResult",
            "comment": null, "successcodes": null, "errorcodes": null,
            "alias": null, "api": null, "deprecated": null,
            "cmdbufferlevel": null, "pipeline": null, "queues": null,
            "renderpass": null, "videocoding": null,
            "raw_content": "", "is_alias": false,
            "parameters": [
              {"name":"pCreateInfo","type_name":"VkInstanceCreateInfo","optional":null,"len":null,
               "altlen":null,"externsync":null,"noautovalidity":null,"objecttype":null,"stride":null,
               "validstructs":null,"api":null,"deprecated":null,"comment":null,
               "definition":"const VkInstanceCreateInfo * pCreateInfo","raw_content":""}
            ]
          }
        ]"#;
        std::fs::write(dir.path().join("functions.json"), json).unwrap();

        let g = StructGenerator::new();
        let set = g.build_driver_written_structs(dir.path());

        assert!(
            set.contains("VkCooperativeVectorPropertiesNV"),
            "a non-const pointer output param must mark its struct driver-written; got {set:?}"
        );
        // A `const` input must not be swept in: doing so would retype fields the
        // *caller* fills, which is a gratuitous API break and not a soundness fix.
        assert!(
            !set.contains("VkInstanceCreateInfo"),
            "a const input param must not count as driver-written; got {set:?}"
        );
    }

    #[test]
    fn unmarked_struct_in_derived_set_still_gets_raw_fields() {
        let g = StructGenerator::new();
        let s = mk_struct(
            "VkCooperativeVectorPropertiesNV",
            vec![
                mk_member("sType", "VkStructureType", None),
                mk_member("inputType", "VkComponentTypeKHR", None),
            ],
        );
        // NOT returnedonly — exactly the vk.xml situation.
        assert!(s.returnedonly.is_none());

        let mut derived = std::collections::HashSet::new();
        derived.insert("VkCooperativeVectorPropertiesNV".to_string());

        let code = g.generate_struct(
            &s,
            &enum_names(&[]),
            &EnumDefaultMap::new(),
            &known_stypes(&[]),
            &enum_names(&["VkComponentTypeKHR", "VkStructureType"]),
            &derived,
            Path::new("."),
        );
        assert!(
            code.contains("pub inputType: i32,"),
            "unmarked-but-derived struct must still get raw fields, got:
{code}"
        );
    }

    fn mk_member(
        name: &str,
        type_name: &str,
        values: Option<&str>,
    ) -> crate::parser::vk_types::StructMember {
        crate::parser::vk_types::StructMember {
            name: name.to_string(),
            type_name: type_name.to_string(),
            optional: None,
            len: None,
            altlen: None,
            noautovalidity: None,
            values: values.map(|s| s.to_string()),
            limittype: None,
            selector: None,
            selection: None,
            externsync: None,
            objecttype: None,
            deprecated: None,
            comment: None,
            api: None,
            definition: String::new(),
            raw_content: String::new(),
        }
    }

    fn mk_struct(
        name: &str,
        members: Vec<crate::parser::vk_types::StructMember>,
    ) -> StructDefinition {
        StructDefinition {
            name: name.to_string(),
            category: "struct".to_string(),
            structextends: None,
            returnedonly: None,
            comment: None,
            allowduplicate: None,
            deprecated: None,
            alias: None,
            api: None,
            members,
            raw_content: String::new(),
            is_alias: false,
            source_line: None,
        }
    }

    fn known_stypes(names: &[&str]) -> KnownStructureTypes {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn enum_names(names: &[&str]) -> std::collections::HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// A struct the *driver* fills must not type an enum field as a Rust enum:
    /// an implementation newer than this `vk.xml` can report a value outside
    /// the declared set, and reading an out-of-range discriminant is UB. The
    /// hazard is reachable by upgrading a graphics driver, with no application
    /// change and no error to observe, so it has to be structurally impossible
    /// rather than merely unlikely.
    #[test]
    fn driver_written_enum_fields_are_emitted_as_raw_integers() {
        let g = StructGenerator::new();
        let mut s = mk_struct(
            "VkFooPropertiesKHR",
            vec![
                mk_member("sType", "VkStructureType", None),
                mk_member("componentType", "VkComponentTypeKHR", None),
                mk_member("count", "uint32_t", None),
            ],
        );
        s.returnedonly = Some("true".to_string());

        let code = g.generate_struct(
            &s,
            &enum_names(&[]),
            &EnumDefaultMap::new(),
            &known_stypes(&[]),
            &enum_names(&["VkComponentTypeKHR", "VkStructureType"]),
            &enum_names(&[]),
            Path::new("."),
        );

        assert!(
            code.contains("pub componentType: i32,"),
            "driver-written enum field must be raw i32, got:\n{code}"
        );
        assert!(
            !code.contains("pub componentType: VkComponentTypeKHR,"),
            "driver-written enum field must not keep its enum type:\n{code}"
        );
        // sType is written by the *application* even in a returnedonly struct —
        // that is how a pNext query chain is assembled — so it keeps its enum.
        assert!(
            code.contains("pub sType: VkStructureType,"),
            "sType is app-written and must keep its enum type:\n{code}"
        );
        // Non-enum fields are untouched.
        assert!(
            code.contains("pub count: u32,"),
            "non-enum fields must be unaffected:\n{code}"
        );
    }

    /// The conversion is scoped to driver-written structs. Application-written
    /// structs keep ergonomic enums, which is most of the surface — the app
    /// only ever writes values it got from the enum, so no invalid discriminant
    /// can arise there.
    #[test]
    fn application_written_structs_keep_their_enum_types() {
        let g = StructGenerator::new();
        let s = mk_struct(
            "VkFooCreateInfoKHR",
            vec![
                mk_member("sType", "VkStructureType", None),
                mk_member("format", "VkFormat", None),
            ],
        );
        assert_eq!(s.returnedonly, None, "fixture must be app-written");

        let code = g.generate_struct(
            &s,
            &enum_names(&[]),
            &EnumDefaultMap::new(),
            &known_stypes(&[]),
            &enum_names(&["VkFormat", "VkStructureType"]),
            &enum_names(&[]),
            Path::new("."),
        );

        assert!(
            code.contains("pub format: VkFormat,"),
            "app-written enum field must keep its enum type:\n{code}"
        );
        assert!(
            !code.contains("pub format: i32,"),
            "app-written field must not be widened to i32:\n{code}"
        );
    }

    /// A driver-written field that is now `i32` must default to `0`, not to an
    /// enum variant path — `enum_defaults` exists to avoid `mem::zeroed()`
    /// producing an invalid discriminant, and that problem cannot arise for an
    /// integer. Emitting `VkComponentTypeKHR::FOO` for an `i32` field would not
    /// compile, so this pins the interaction between the two mechanisms.
    #[test]
    fn driver_written_field_defaults_to_zero_not_a_variant_path() {
        let g = StructGenerator::new();
        let mut s = mk_struct(
            "VkBarPropertiesKHR",
            vec![mk_member("componentType", "VkComponentTypeKHR", None)],
        );
        s.returnedonly = Some("true".to_string());

        let mut defaults = EnumDefaultMap::new();
        defaults.insert(
            "VkComponentTypeKHR".to_string(),
            "COMPONENT_TYPE_FLOAT16_KHR".to_string(),
        );

        let code = g.generate_struct(
            &s,
            &enum_names(&[]),
            &defaults,
            &known_stypes(&[]),
            &enum_names(&["VkComponentTypeKHR"]),
            &enum_names(&[]),
            Path::new("."),
        );

        assert!(
            code.contains("componentType: 0,"),
            "an i32 field must default to 0:\n{code}"
        );
        assert!(
            !code.contains("componentType: VkComponentTypeKHR::"),
            "must not emit a variant path as the default for an i32 field:\n{code}"
        );
    }

    #[test]
    fn emits_pnext_impl_for_chain_header_struct() {
        let g = StructGenerator::new();
        let s = mk_struct(
            "VkPhysicalDeviceFooFeaturesKHR",
            vec![
                mk_member(
                    "sType",
                    "VkStructureType",
                    Some("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FOO_FEATURES_KHR"),
                ),
                mk_member("pNext", "void", None),
                mk_member("foo", "VkBool32", None),
            ],
        );
        let known = known_stypes(&["VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FOO_FEATURES_KHR"]);
        let impl_code = g
            .try_emit_pnext_chainable_impl(&s, &known)
            .expect("should emit impl");
        assert!(impl_code.contains(
            "unsafe impl super::pnext::PNextChainable for VkPhysicalDeviceFooFeaturesKHR"
        ));
        assert!(
            impl_code.contains("VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_FOO_FEATURES_KHR")
        );
    }

    #[test]
    fn skips_structs_without_chain_header() {
        let g = StructGenerator::new();
        let s = mk_struct(
            "VkExtent2D",
            vec![
                mk_member("width", "uint32_t", None),
                mk_member("height", "uint32_t", None),
            ],
        );
        assert!(
            g.try_emit_pnext_chainable_impl(&s, &known_stypes(&[]))
                .is_none()
        );
    }

    #[test]
    fn skips_base_structures_with_no_fixed_stype() {
        let g = StructGenerator::new();
        let s = mk_struct(
            "VkBaseOutStructure",
            vec![
                mk_member("sType", "VkStructureType", None),
                mk_member("pNext", "void", None),
            ],
        );
        assert!(
            g.try_emit_pnext_chainable_impl(&s, &known_stypes(&[]))
                .is_none()
        );
    }

    #[test]
    fn skips_unions_and_aliases() {
        let g = StructGenerator::new();
        let known = known_stypes(&["VK_STRUCTURE_TYPE_APPLICATION_INFO"]);
        let mut s = mk_struct(
            "VkWeirdUnion",
            vec![
                mk_member(
                    "sType",
                    "VkStructureType",
                    Some("VK_STRUCTURE_TYPE_APPLICATION_INFO"),
                ),
                mk_member("pNext", "void", None),
            ],
        );
        s.category = "union".to_string();
        assert!(g.try_emit_pnext_chainable_impl(&s, &known).is_none());
        s.category = "struct".to_string();
        s.is_alias = true;
        assert!(g.try_emit_pnext_chainable_impl(&s, &known).is_none());
    }

    #[test]
    fn skips_struct_whose_stype_is_not_emitted() {
        // Mirrors the Vulkan-SC case: the struct is defined but its sType
        // enum variant is absent from VkStructureType, so we must skip.
        let g = StructGenerator::new();
        let s = mk_struct(
            "VkDeviceObjectReservationCreateInfo",
            vec![
                mk_member(
                    "sType",
                    "VkStructureType",
                    Some("VK_STRUCTURE_TYPE_DEVICE_OBJECT_RESERVATION_CREATE_INFO"),
                ),
                mk_member("pNext", "void", None),
            ],
        );
        // known_stypes does NOT contain the variant.
        assert!(
            g.try_emit_pnext_chainable_impl(
                &s,
                &known_stypes(&["VK_STRUCTURE_TYPE_APPLICATION_INFO"])
            )
            .is_none()
        );
    }

    #[test]
    fn test_default_values() {
        let generator = StructGenerator::new();

        // Integer primitives default to 0
        assert_eq!(generator.get_default_value_for_rust_type("u32", false), "0");
        // Float primitives default to 0.0 (NOT 0 — that would be a type error)
        assert_eq!(
            generator.get_default_value_for_rust_type("f32", false),
            "0.0"
        );
        // Const pointers default to null
        assert_eq!(
            generator.get_default_value_for_rust_type("*const u32", true),
            "std::ptr::null()"
        );
        // Mut pointers default to null_mut
        assert_eq!(
            generator.get_default_value_for_rust_type("*mut u32", true),
            "std::ptr::null_mut()"
        );
        // Unknown types (Vulkan handles, structs) fall back to zeroed for FFI safety
        assert_eq!(
            generator.get_default_value_for_rust_type("VkDevice", false),
            "unsafe { std::mem::zeroed() }"
        );
        // Arrays use zeroed because element types may be complex
        assert_eq!(
            generator.get_default_value_for_rust_type("[c_char; 256]", false),
            "unsafe { std::mem::zeroed() }"
        );
    }
}
