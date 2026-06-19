//! Type generator module
//!
//! Generates Rust type aliases from types.json intermediate file

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorModule, GeneratorResult};

use crate::parser::vk_types::TypeDefinition;

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

/// Generator module for Vulkan type aliases
pub struct TypeGenerator;

impl Default for TypeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Public wrapper for generate_type, used by FunctionGenerator for funcpointer types
    pub fn generate_type_public(
        &self,
        type_def: &TypeDefinition,
        all_type_names: &std::collections::HashSet<String>,
        output_dir: &Path,
    ) -> String {
        self.generate_type(type_def, all_type_names, output_dir)
    }

    /// Generate code for a single type alias
    fn generate_type(
        &self,
        type_def: &TypeDefinition,
        _all_type_names: &std::collections::HashSet<String>,
        _output_dir: &Path,
    ) -> String {
        let mut code = String::new();
        let sanitized_name = sanitize_type_name(&type_def.name);

        // If this entry actually represents a C preprocessor define (many
        // intermediate files include '#define'/'#if' blocks under types.json),
        // skip emitting a Rust `pub type` here. The macros generator is
        // responsible for turning complex defines/conditionals into valid Rust
        // constructs. Emitting a `pub type` from a raw preprocessor token often
        // produced invalid tokens like `endif` or `object` as types.
        if type_def.category == "define" {
            code.push_str(&format!("// Skipped define `{}`\n\n", type_def.name));

            return code;
        }

        // Add documentation (skip multi-line raw content with preprocessor directives)
        if let Some(comment) = &type_def.comment {
            for line in comment.lines() {
                code.push_str(&format!(
                    "/// {}\n",
                    crate::codegen::sanitize_doc_line(line)
                ));
            }
        }
        let raw = type_def.raw_content.trim();
        if !raw.is_empty() && !raw.contains('#') && !raw.contains('@') {
            code.push_str("/// \n");
            for line in raw.lines() {
                code.push_str(&format!(
                    "/// From vk.xml: {}\n",
                    crate::codegen::sanitize_doc_line(line)
                ));
            }
        }

        // Note: api="vulkan"/"vulkansc" attributes are not emitted as #[cfg] since
        // we generate comprehensive bindings for all of Vulkan.

        // Add deprecation warnings
        if let Some(deprecated) = &type_def.deprecated {
            if deprecated == "true" {
                code.push_str("#[deprecated]\n");
            } else {
                code.push_str(&format!("#[deprecated(note = \"{}\")]\n", deprecated));
            }
        }

        // Handle alias types
        if type_def.is_alias {
            if let Some(alias_target) = &type_def.alias {
                // Normalize alias targets: strip stray 'const' prefixes or
                // other accidental tokens and extract the trailing identifier.
                // This handles malformed inputs like "constVkSampler" or
                // "const VkSampler" that sometimes appear in intermediate
                // files.
                // Helper to aggressively clean noisy type tokens. This handles
                // joined prefixes like "constVkSampler", spaced forms like
                // "const VkSampler", stray "struct" tokens, and removes
                // non-identifier characters. It prefers the trailing identifier
                // (last word) when possible.
                fn clean_typename(raw: &str) -> String {
                    use regex::Regex;

                    if raw.trim().is_empty() {
                        return String::new();
                    }

                    // If the token begins with a concatenated 'const' (e.g.
                    // "constVkSampler"), strip that prefix first.
                    let mut s = raw.to_string();
                    if let Ok(pref_re) = Regex::new(r"^const(?P<rest>.+)") {
                        if let Some(cap) = pref_re.captures(&s) {
                            if let Some(rest) = cap.name("rest") {
                                s = rest.as_str().to_string();
                            }
                        }
                    }

                    // Remove stray 'const' or 'struct' tokens, replace non-id
                    // characters with spaces, collapse whitespace, then trim.
                    s = s.replace("const", "");
                    s = s.replace("struct", "");
                    // Replace any non-alphanumeric/underscore with a space
                    let non_id = Regex::new(r"[^A-Za-z0-9_]+").unwrap();
                    s = non_id.replace_all(&s, " ").to_string();
                    let ws = Regex::new(r"\s+").unwrap();
                    s = ws.replace_all(&s, " ").to_string();
                    s = s.trim().to_string();

                    // Prefer trailing identifier if present
                    if let Ok(id_re) = Regex::new(r"([A-Za-z_][A-Za-z0-9_]*)$") {
                        if let Some(cap) = id_re.captures(&s) {
                            return cap[1].to_string();
                        }
                    }

                    s
                }

                let resolved_target = clean_typename(alias_target.as_str());

                // If the resolved target still isn't known, try Flags->FlagBits
                // canonicalization on both the cleaned and original forms.
                let mut found: Option<String> = None;
                // Build a prioritized list of candidate names to try resolving.
                let mut candidates = vec![resolved_target.clone(), alias_target.clone()];
                // Also try a leading-const-stripped original if present (e.g.
                // alias_target == "constVkSampler") to help resolution.
                if alias_target.starts_with("const") {
                    let lc = alias_target.trim_start_matches("const").to_string();
                    if !lc.is_empty() {
                        candidates.push(lc);
                    }
                }
                // Try candidate forms
                for cand in &candidates {
                    if _all_type_names.contains(cand.as_str()) {
                        found = Some(cand.clone());
                        break;
                    }
                    if cand.ends_with("Flags") {
                        let candidate = cand.replace("Flags", "FlagBits");
                        if _all_type_names.contains(candidate.as_str()) {
                            found = Some(candidate.clone());
                            break;
                        }
                    }
                }

                // Fallback try: if resolved_target still looks like it begins with a
                // lowercase run before 'Vk', strip until 'Vk' and try again.
                if found.is_none() {
                    if let Some(pos) = resolved_target.find("Vk") {
                        let stripped = resolved_target[pos..].to_string();
                        if _all_type_names.contains(stripped.as_str()) {
                            found = Some(stripped);
                        }
                    }
                }

                // When a candidate is found, prefer a cleaned/canonical
                // form without stray tokens like leading 'const' or
                // 'struct'. This prevents emitting types such as
                // 'constVkSampler' into the final Rust file when a
                // canonical 'VkSampler' exists.
                let final_target = if let Some(f) = found {
                    // Aggressively clean common noise from the chosen
                    // candidate, then prefer the cleaned variant if it
                    // exists in the known type set. Fall back to the
                    // original found value otherwise.
                    let cleaned = clean_typename(&f);

                    // Prefer the cleaned trailing identifier if present
                    if !_all_type_names.is_empty() && _all_type_names.contains(cleaned.as_str()) {
                        sanitize_type_name(&cleaned)
                    } else {
                        sanitize_type_name(&f)
                    }
                } else {
                    // Use u32 as a safe default for bitmask-style aliases
                    String::from("u32")
                };
                // Ensure we don't accidentally emit names with leading
                // 'const'/'struct' noise — aggressively strip those
                // patterns from the resolved target before emitting.
                fn strip_leading_const_struct(s: &str) -> String {
                    use regex::Regex;
                    if s.trim().is_empty() {
                        return String::new();
                    }
                    // Match repeated leading 'const' or 'struct' possibly joined
                    // directly to the identifier (e.g. 'constVkFoo') or
                    // separated by underscores/spaces.
                    if let Ok(re) = Regex::new(r"^(?:(?:const|struct)[_ ]?)+(?P<rest>.+)$") {
                        if let Some(cap) = re.captures(s) {
                            if let Some(rest) = cap.name("rest") {
                                return rest.as_str().to_string();
                            }
                        }
                    }
                    s.to_string()
                }

                let mut emitted_target = strip_leading_const_struct(&final_target);
                // Also collapse accidental spaces/invalid chars into a safe id
                emitted_target = sanitize_type_name(&emitted_target);

                if emitted_target.is_empty() {
                    emitted_target = "u32".to_string();
                }

                code.push_str(&format!(
                    "pub type {} = {};\n\n",
                    sanitized_name, emitted_target
                ));
                return code;
            }
        }

        // Handle different type categories
        match type_def.category.as_str() {
            "basetype" => {
                // Skip basetypes with preprocessor directives (Objective-C types)
                if type_def.definition.as_deref().is_some_and(|d| {
                    d.contains("#ifdef")
                        || d.contains("#if")
                        || d.contains("@class")
                        || d.contains("@protocol")
                }) {
                    code.push_str(&format!(
                        "// Skipped platform-specific basetype `{}`\n",
                        type_def.name
                    ));
                    code.push_str(&format!("pub type {} = *mut c_void;\n\n", sanitized_name));
                    return code;
                }

                // Use type_references to determine the Rust type
                let rust_type = if !type_def.type_references.is_empty() {
                    let base = &type_def.type_references[0];
                    let mapped = self.map_type_to_rust(base);
                    // Check if the definition involves a pointer
                    let is_ptr = type_def
                        .definition
                        .as_deref()
                        .is_some_and(|d| d.contains('*'));
                    if is_ptr {
                        format!("*mut {}", mapped)
                    } else {
                        mapped
                    }
                } else if type_def
                    .definition
                    .as_deref()
                    .is_some_and(|d| d.contains("struct"))
                {
                    // Opaque struct declaration (e.g., struct ANativeWindow;)
                    "*mut c_void".to_string()
                } else {
                    // Fallback to opaque type
                    "*mut c_void".to_string()
                };
                code.push_str(&format!("pub type {} = {};\n\n", sanitized_name, rust_type));
            }
            "handle" => {
                // Dispatchable handles are pointers, non-dispatchable are u64
                let is_non_dispatchable = type_def
                    .definition
                    .as_deref()
                    .or(Some(&type_def.raw_content))
                    .is_some_and(|d| d.contains("NON_DISPATCHABLE"));
                if is_non_dispatchable {
                    code.push_str(&format!("pub type {} = u64;\n\n", sanitized_name));
                } else {
                    code.push_str(&format!("pub type {} = *mut c_void;\n\n", sanitized_name));
                }
            }
            "bitmask" => {
                // Check definition for VkFlags64 vs VkFlags (u64 vs u32)
                let is_64 = type_def
                    .definition
                    .as_deref()
                    .or(Some(&type_def.raw_content))
                    .is_some_and(|d| d.contains("VkFlags64") || d.contains("uint64_t"));
                let rust_type = if is_64 { "u64" } else { "u32" };
                code.push_str(&format!("pub type {} = {};\n\n", sanitized_name, rust_type));
            }
            "funcpointer" => {
                // Function pointers
                if let Some(definition) = &type_def.definition {
                    let rust_fn_type = self.parse_function_pointer(definition);
                    code.push_str(&format!(
                        "pub type {} = {};\n\n",
                        sanitized_name, rust_fn_type
                    ));
                } else {
                    code.push_str(&format!("pub type {} = *const c_void;\n\n", sanitized_name));
                }
            }
            "enum" => {
                // Enum types are handled by the enum generator, skip here
                return String::new();
            }
            "include" | "define" => {
                // Already handled above
                return String::new();
            }
            // External types with no category — these come from
            // platform headers (X11/Xlib.h, xcb/xcb.h, windows.h, …).
            // vk.xml only carries the requires="header" attribute, not
            // the actual C type, so we hard-map the well-known X11 /
            // XCB integer ID types to their canonical widths. Anything
            // else falls back to *mut c_void (the safe default for
            // pointer-shaped opaque platform handles like Display* /
            // wl_display* / xcb_connection_t* / HWND).
            //
            // A "" category with no `requires` carries no type info at all,
            // so it falls through to the `_` arm below and emits nothing.
            "" if type_def.requires.is_some() => {
                let rust_type = match sanitized_name.as_str() {
                    // Xlib XIDs are `unsigned long` — 32-bit on
                    // Win32, 64-bit on Linux x86_64. c_ulong matches.
                    "Window" | "VisualID" | "RROutput" => "c_ulong",
                    // XCB IDs are uint32_t.
                    "xcb_window_t" | "xcb_visualid_t" => "u32",
                    // Direct-to-display.
                    "zx_handle_t" => "u32",
                    // Everything else is a pointer-shaped opaque
                    // platform handle.
                    _ => "*mut c_void",
                };
                code.push_str(&format!("pub type {} = {};\n\n", sanitized_name, rust_type));
            }
            _ => {
                // Skip unknown categories (including "" with no `requires`).
                return String::new();
            }
        }

        code
    }

    /// Map Vulkan types to Rust types
    fn map_type_to_rust(&self, vulkan_type: &str) -> String {
        match vulkan_type {
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

    /// Parse a function pointer definition
    fn parse_function_pointer(&self, _definition: &str) -> String {
        // Heuristic parser for typedef-style function-pointer definitions.
        // Examples handled:
        //  typedef void (VKAPI_PTR *PFN_name)(int a, const char* b);
        //  void (VKAPI_PTR *PFN_name)(void);
        // The goal is to emit an Option<unsafe extern "C" fn(...) -> Ret>
        use regex::Regex;

        let mut def = _definition.to_string();

        // Remove XML tags if present
        let tag_re = Regex::new(r#"<[^>]+>"#).unwrap();
        def = tag_re.replace_all(&def, " ").to_string();

        // Collapse whitespace
        let ws = Regex::new(r"\s+").unwrap();
        def = ws.replace_all(&def, " ").to_string();
        let def = def.trim();

        // Regex to capture: typedef <ret> (<convention>* <name>) (<params>)
        // We're generous with spaces and tokens; use (?s) to allow multiline params
        let fp_re = Regex::new(r"(?s)typedef\s+(?P<ret>[^()]+?)\s*\([^)]*?\*\s*(?P<name>[A-Za-z0-9_]+)\)\s*\((?P<params>.*)\)\s*;?")
            .or_else(|_| Regex::new(r"(?s)(?P<ret>[^()]+?)\s*\([^)]*?\*\s*(?P<name>[A-Za-z0-9_]+)\)\s*\((?P<params>.*)\)\s*;?"))
            .unwrap();

        if let Some(cap) = fp_re.captures(def) {
            let ret = cap.name("ret").map(|m| m.as_str().trim()).unwrap_or("void");
            let params = cap.name("params").map(|m| m.as_str()).unwrap_or("");

            // Parse return type
            let (ret_base, _ret_const, _ret_ptr) = Self::analyze_type_tokens(ret);
            let rust_ret = self.map_type_to_rust(&ret_base);

            // Parse parameters. Hoist the identifier regex out of the loop
            // so we don't recompile it on every parameter.
            let ident_re = Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").unwrap();
            let mut rust_params = Vec::new();
            let mut arg_index: usize = 0;
            for raw_p in params.split(',') {
                let p = raw_p.trim();
                if p.is_empty() || p == "void" {
                    continue;
                }

                // Split tokens and try to separate the name (last token) from the type
                let toks: Vec<&str> = p.split_whitespace().collect();
                let mut maybe_name: Option<&str> = None;
                let mut type_tokens: Vec<&str> = toks.clone();
                if toks.len() >= 2 {
                    let last = toks[toks.len() - 1];
                    // If the last token looks like an identifier (param name), treat as name
                    if ident_re.is_match(last) {
                        maybe_name = Some(last);
                        type_tokens = toks[..toks.len() - 1].to_vec();
                    }
                }

                let type_str = type_tokens.join(" ");
                let (base, is_const, is_ptr) = Self::analyze_type_tokens(&type_str);

                let mut rust_ty = self.map_type_to_rust(&base);
                if is_ptr {
                    if is_const {
                        rust_ty = format!("*const {}", rust_ty);
                    } else {
                        rust_ty = format!("*mut {}", rust_ty);
                    }
                }

                let arg_name = if let Some(n) = maybe_name {
                    // sanitize arg name
                    n.to_string()
                } else {
                    arg_index += 1;
                    format!("arg{}", arg_index)
                };

                rust_params.push(format!("{}: {}", arg_name, rust_ty));
            }

            let params_joined = rust_params.join(", ");

            // Return type formatting: void becomes () in Rust
            let ret_rust = if rust_ret == "c_void" {
                "()".to_string()
            } else {
                rust_ret
            };

            // Emit Option<unsafe extern "C" fn(...) -> Ret>
            if params_joined.is_empty() {
                return format!("Option<unsafe extern \"system\" fn() -> {}>", ret_rust);
            }
            return format!(
                "Option<unsafe extern \"system\" fn({}) -> {}>",
                params_joined, ret_rust
            );
        }

        // Fallback: generic function pointer option
        "Option<unsafe extern \"system\" fn()>".to_string()
    }

    /// Helper to analyze a C-style type token string and return (base_type, is_const, is_pointer)
    fn analyze_type_tokens(t: &str) -> (String, bool, bool) {
        let s = t.replace('\u{00A0}', " "); // non-breaking spaces
        let s = s.replace('*', " * ");
        let parts: Vec<&str> = s.split_whitespace().collect();

        let mut is_const = false;
        let mut is_ptr = false;
        let mut base = String::new();

        for part in parts {
            if part == "const" {
                is_const = true;
            } else if part == "*" {
                is_ptr = true;
            } else if part == "VKAPI_PTR" || part == "APIENTRY" || part == "CALLBACK" {
                // ignore calling convention tokens
            } else {
                // treat as base type token
                base = part.to_string();
            }
        }

        if base.is_empty() {
            base = "void".to_string();
        }

        (base, is_const, is_ptr)
    }

    /// Generate all type aliases from input directory
    fn generate_all_types(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("types.json");
        let input_content = fs::read_to_string(&input_path).map_err(GeneratorError::Io)?;

        // Parse JSON - try array first, then object-with-array { "types": [...] }
        let types: Vec<TypeDefinition> =
            match serde_json::from_str::<Vec<TypeDefinition>>(&input_content) {
                Ok(v) => v,
                Err(_) => {
                    #[derive(serde::Deserialize)]
                    struct TypesFile {
                        types: Vec<TypeDefinition>,
                    }

                    let wrapper: TypesFile =
                        serde_json::from_str(&input_content).map_err(GeneratorError::Json)?;
                    wrapper.types
                }
            };

        // Generate code
        let mut generated_code = String::new();

        // Don't add imports here - they're handled by the assembler

        // Add allow directives
        generated_code.push_str("#[allow(non_camel_case_types)]\n");
        generated_code.push_str("#[allow(dead_code)]\n\n");

        // Determine enum names so we don't emit type aliases that conflict
        // with explicit enum definitions (avoid duplicate type/enum definitions).
        let mut enum_names = std::collections::HashSet::new();
        let enums_path = input_dir.join("enums.json");
        if enums_path.exists() {
            if let Ok(enums_content) = fs::read_to_string(&enums_path) {
                if let Ok(enums_array) = serde_json::from_str::<
                    Vec<crate::parser::vk_types::EnumDefinition>,
                >(&enums_content)
                {
                    for e in enums_array {
                        enum_names.insert(e.name);
                    }
                } else {
                    // Try wrapper object { "enums": [...] }
                    #[derive(serde::Deserialize)]
                    struct EnumsFile {
                        enums: Vec<crate::parser::vk_types::EnumDefinition>,
                    }
                    if let Ok(wrapper) = serde_json::from_str::<EnumsFile>(&enums_content) {
                        for e in wrapper.enums {
                            enum_names.insert(e.name);
                        }
                    }
                }
            }
        }

        // Also collect struct names so we don't emit type aliases that conflict
        // with generated structs (avoid type alias vs struct collisions).
        let mut struct_names = std::collections::HashSet::new();
        let structs_path = input_dir.join("structs.json");
        if structs_path.exists() {
            if let Ok(structs_content) = fs::read_to_string(&structs_path) {
                if let Ok(structs_array) = serde_json::from_str::<
                    Vec<crate::parser::vk_types::StructDefinition>,
                >(&structs_content)
                {
                    for s in structs_array {
                        struct_names.insert(s.name);
                    }
                } else {
                    #[derive(serde::Deserialize)]
                    struct StructsFile {
                        structs: Vec<crate::parser::vk_types::StructDefinition>,
                    }
                    if let Ok(wrapper) = serde_json::from_str::<StructsFile>(&structs_content) {
                        for s in wrapper.structs {
                            struct_names.insert(s.name);
                        }
                    }
                }
            }
        }

        // Generate type aliases
        let raw_type_names: std::collections::HashSet<String> =
            types.iter().map(|t| t.name.clone()).collect();

        // Build a normalized view of type names to help resolve noisy
        // intermediate artifacts like "constVkSampler" or "struct VkFoo".
        // We include both the original and cleaned variants so alias
        // resolution can match either form. Also include known enum and
        // struct names so alias targets that reference those definitions
        // (which may come from enums.json/structs.json) are considered
        // during resolution.
        let mut all_type_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        use regex::Regex;
        let id_re = Regex::new(r"([A-Za-z_][A-Za-z0-9_]*)$").unwrap();
        for name in raw_type_names.iter() {
            all_type_names.insert(name.clone());

            // aggressively strip common noise
            let mut cleaned = name.replace("const", "");
            cleaned = cleaned.replace("struct", "");
            cleaned = cleaned.replace(|c: char| c.is_whitespace(), " ");
            cleaned = cleaned.trim().to_string();

            if let Some(cap) = id_re.captures(&cleaned) {
                let trailing = cap[1].to_string();
                if trailing != *name {
                    all_type_names.insert(trailing);
                }
            }

            // If the name contains 'Vk' later (e.g. 'constVkFoo'), try stripping
            // everything up to the 'Vk' prefix and include that as a candidate.
            if let Some(pos) = cleaned.find("Vk") {
                let stripped = cleaned[pos..].to_string();
                if stripped != *name {
                    all_type_names.insert(stripped);
                }
            }
        }

        // Also include explicit enum and struct names as resolution
        // candidates. These may not appear in the raw types.json list
        // but are valid targets for aliases; including them prevents
        // emitting noisy unresolved alias names like 'constVkFoo' when
        // the canonical 'VkFoo' exists as a struct or enum.
        for en in enum_names.iter() {
            all_type_names.insert(en.clone());
            if let Some(cap) = id_re.captures(en) {
                let trailing = cap[1].to_string();
                if trailing != *en {
                    all_type_names.insert(trailing);
                }
            }
        }
        for sn in struct_names.iter() {
            all_type_names.insert(sn.clone());
            if let Some(cap) = id_re.captures(sn) {
                let trailing = cap[1].to_string();
                if trailing != *sn {
                    all_type_names.insert(trailing);
                }
            }
        }

        // No placeholder types needed - the tree parser captures all types correctly
        let placeholder_code = String::new();

        // Group types by category for better organization
        let mut base_types = Vec::new();
        let mut handles = Vec::new();
        let mut bitmasks = Vec::new();
        let mut funcpointers = Vec::new();
        let mut others = Vec::new();

        for type_def in &types {
            match type_def.category.as_str() {
                "basetype" => base_types.push(type_def),
                "handle" => handles.push(type_def),
                "bitmask" => bitmasks.push(type_def),
                "funcpointer" => funcpointers.push(type_def),
                _ => others.push(type_def),
            }
        }

        // Insert placeholders first so later emitted aliases can reference them
        generated_code.push_str(&placeholder_code);

        // Generate in logical order
        if !base_types.is_empty() {
            generated_code.push_str("// Base types\n");
            for type_def in base_types {
                // Avoid emitting a type alias if an enum or struct with the same name exists
                if enum_names.contains(&type_def.name) || struct_names.contains(&type_def.name) {
                    // Emit a comment indicating we skipped the conflicting type
                    generated_code.push_str(&format!(
                        "// Skipped type alias for {} because an enum or struct with that name exists\n\n",
                        type_def.name
                    ));
                    continue;
                }
                generated_code.push_str(&self.generate_type(type_def, &all_type_names, output_dir));
            }
        }

        if !handles.is_empty() {
            generated_code.push_str("// Handle types\n");
            for type_def in handles {
                if enum_names.contains(&type_def.name) || struct_names.contains(&type_def.name) {
                    generated_code.push_str(&format!(
                        "// Skipped handle alias for {} because an enum or struct with that name exists\n\n",
                        type_def.name
                    ));
                    continue;
                }
                generated_code.push_str(&self.generate_type(type_def, &all_type_names, output_dir));
            }
        }

        if !bitmasks.is_empty() {
            generated_code.push_str("// Bitmask types\n");
            for type_def in bitmasks {
                if enum_names.contains(&type_def.name) || struct_names.contains(&type_def.name) {
                    generated_code.push_str(&format!(
                        "// Skipped bitmask alias for {} because an enum or struct with that name exists\n\n",
                        type_def.name
                    ));
                    continue;
                }
                generated_code.push_str(&self.generate_type(type_def, &all_type_names, output_dir));
            }
        }

        // Funcpointer types are emitted by the FunctionGenerator (after structs)
        // to avoid forward reference issues with struct types used as parameters.

        if !others.is_empty() {
            generated_code.push_str("// Other types\n");
            for type_def in others {
                if enum_names.contains(&type_def.name) || struct_names.contains(&type_def.name) {
                    generated_code.push_str(&format!(
                        "// Skipped other alias for {} because an enum or struct with that name exists\n\n",
                        type_def.name
                    ));
                    continue;
                }
                generated_code.push_str(&self.generate_type(type_def, &all_type_names, output_dir));
            }
        }

        // Ensure output directory exists
        fs::create_dir_all(output_dir).map_err(GeneratorError::Io)?;

        // Write output file
        let output_path = output_dir.join(self.output_file());
        fs::write(output_path, generated_code).map_err(GeneratorError::Io)?;

        crate::codegen::logging::log_info(&format!(
            "TypeGeneratorModule: Generated {} type aliases",
            types.len()
        ));
        Ok(())
    }
}

impl GeneratorModule for TypeGenerator {
    fn name(&self) -> &str {
        "TypeGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["types.json".to_string()]
    }

    fn output_file(&self) -> String {
        "types.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        Vec::new() // Type aliases don't depend on other modules typically
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        self.generate_all_types(input_dir, output_dir)?;
        Ok(())
    }
}
