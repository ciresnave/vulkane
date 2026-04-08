//! Macros generator module (FIXED VERSION)
//!
//! Generates Rust macro definitions from macros.json intermediate file

use crate::parser::vk_types::MacroDefinition;
use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorMetadata, GeneratorModule, GeneratorResult};

/// Generator module for Vulkan macros (Fixed Version)
pub struct MacroGenerator;

impl Default for MacroGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl MacroGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Rust code for a single macro
    fn generate_macro(&self, macro_def: &MacroDefinition) -> String {
        // Skip macros that do not have a valid name (defensive)
        if macro_def.name.trim().is_empty() {
            return String::new();
        }

        // Skip Vulkan SC (Safety Critical) macros — vk.xml ships
        // VKSC_API_VARIANT and VKSC_API_VERSION_1_0 unconditionally
        // (no `api` attribute we can filter on at the parser level),
        // but desktop Vulkan apps cannot use them. The clean solution
        // is a name-based filter at the codegen entry point.
        if macro_def.name.starts_with("VKSC_") {
            return String::new();
        }

        let mut code = String::new();

        // Clean HTML entities from the definition first
        let definition_str = macro_def.definition.as_str();
        let cleaned_definition = self.clean_html_entities(definition_str);

        match macro_def.macro_type.as_str() {
            "function_like" => {
                code.push_str(&self.generate_function_like_macro(macro_def, &cleaned_definition));
            }
            "object_like" => {
                code.push_str(&self.generate_object_like_macro(macro_def, &cleaned_definition));
            }
            "conditional" => {
                code.push_str(&self.generate_conditional_macro(macro_def, &cleaned_definition));
            }
            _ => {
                // Generic macro handling
                code.push_str(&self.generate_generic_macro(macro_def, &cleaned_definition));
            }
        }

        code.push('\n');
        code
    }

    /// Generate function-like macro as Rust const fn or macro
    fn generate_function_like_macro(
        &self,
        macro_def: &MacroDefinition,
        cleaned_definition: &str,
    ) -> String {
        let mut code = String::new();

        // Documentation
        code.push_str(&format!(
            "// Vulkan macro: {} (from line {})\n",
            macro_def.name,
            self.fmt_line(macro_def.source_line)
        ));

        // If this "function-like" macro is actually a constant definition
        // that invokes another macro (e.g., VK_API_VERSION_1_0 = VK_MAKE_API_VERSION(0,1,0,0)),
        // try to evaluate it as a constant first.
        if let Some(const_def) = self.try_convert_to_constant(&macro_def.name, cleaned_definition) {
            code.push_str(&const_def);
            return code;
        }

        // Try to transpile the C macro body to a Rust const fn.
        // This handles version macros, extraction macros, and any other
        // function-like macros whose bodies use only bitwise ops and casts.
        if !macro_def.parameters.is_empty() {
            if let Some(rust_fn) = self.try_transpile_c_macro(
                &macro_def.name,
                &macro_def.parameters,
                cleaned_definition,
                macro_def.deprecated.as_deref(),
            ) {
                code.push_str(&rust_fn);
                return code;
            }
        }

        // For macros with complex preprocessor conditionals, try to extract
        // the simplest unconditional #define line as a constant value.
        if let Some(const_def) =
            self.try_extract_fallback_define(&macro_def.name, cleaned_definition)
        {
            code.push_str(&const_def);
            return code;
        }

        // Skip internal implementation macros that define handle types
        match macro_def.name.as_str() {
            "VK_DEFINE_HANDLE" | "VK_DEFINE_NON_DISPATCHABLE_HANDLE" => {
                return code;
            }
            _ => {
                // Generic function-like macro fallback - properly comment out C code
                code.push_str(&format!(
                    "// TODO: Implement function-like macro {} manually\n",
                    macro_def.name
                ));

                // Comment out each line of the original C definition
                for line in cleaned_definition.lines() {
                    code.push_str(&format!("// {}\n", line));
                }
            }
        }

        code
    }

    /// Generate object-like macro as const
    fn generate_object_like_macro(
        &self,
        macro_def: &MacroDefinition,
        cleaned_definition: &str,
    ) -> String {
        let mut code = String::new();

        // Documentation
        code.push_str(&format!(
            "// Vulkan macro: {} (from line {})\n",
            macro_def.name,
            self.fmt_line(macro_def.source_line)
        ));

        // Special handling for important Vulkan constants and function-like macros
        match macro_def.name.as_str() {
            // Handle deprecated version macros that should be function-like
            "VK_MAKE_VERSION" => {
                code.push_str("// DEPRECATED: This define is deprecated. VK_MAKE_API_VERSION should be used instead.\n");
                code.push_str(
                    "pub const fn vk_make_version(major: u32, minor: u32, patch: u32) -> u32 {\n\
                        (major << 22) | (minor << 12) | patch\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_MAKE_VERSION(major: u32, minor: u32, patch: u32) -> u32 {\n\
                        vk_make_version(major, minor, patch)\n\
                    }",
                );
            }
            "VK_VERSION_MAJOR" => {
                code.push_str("// DEPRECATED: This define is deprecated. VK_API_VERSION_MAJOR should be used instead.\n");
                code.push_str(
                    "pub const fn vk_version_major(version: u32) -> u32 {\n\
                        version >> 22\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_VERSION_MAJOR(version: u32) -> u32 {\n\
                        vk_version_major(version)\n\
                    }",
                );
            }
            "VK_VERSION_MINOR" => {
                code.push_str("// DEPRECATED: This define is deprecated. VK_API_VERSION_MINOR should be used instead.\n");
                code.push_str(
                    "pub const fn vk_version_minor(version: u32) -> u32 {\n\
                        (version >> 12) & 0x3FF\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_VERSION_MINOR(version: u32) -> u32 {\n\
                        vk_version_minor(version)\n\
                    }",
                );
            }
            "VK_VERSION_PATCH" => {
                code.push_str("// DEPRECATED: This define is deprecated. VK_API_VERSION_PATCH should be used instead.\n");
                code.push_str(
                    "pub const fn vk_version_patch(version: u32) -> u32 {\n\
                        version & 0xFFF\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_VERSION_PATCH(version: u32) -> u32 {\n\
                        vk_version_patch(version)\n\
                    }",
                );
            }
            "VK_MAKE_API_VERSION" => {
                code.push_str(
                    "pub const fn vk_make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {\n\
                        (variant << 29) | (major << 22) | (minor << 12) | patch\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_MAKE_API_VERSION(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {\n\
                        vk_make_api_version(variant, major, minor, patch)\n\
                    }"
                );
            }
            "VK_API_VERSION_VARIANT" => {
                code.push_str(
                    "pub const fn vk_api_version_variant(version: u32) -> u32 {\n\
                        version >> 29\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_API_VERSION_VARIANT(version: u32) -> u32 {\n\
                        vk_api_version_variant(version)\n\
                    }",
                );
            }
            "VK_API_VERSION_MAJOR" => {
                code.push_str(
                    "pub const fn vk_api_version_major(version: u32) -> u32 {\n\
                        (version >> 22) & 0x7F\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_API_VERSION_MAJOR(version: u32) -> u32 {\n\
                        vk_api_version_major(version)\n\
                    }",
                );
            }
            "VK_API_VERSION_MINOR" => {
                code.push_str(
                    "pub const fn vk_api_version_minor(version: u32) -> u32 {\n\
                        (version >> 12) & 0x3FF\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_API_VERSION_MINOR(version: u32) -> u32 {\n\
                        vk_api_version_minor(version)\n\
                    }",
                );
            }
            "VK_API_VERSION_PATCH" => {
                code.push_str(
                    "pub const fn vk_api_version_patch(version: u32) -> u32 {\n\
                        version & 0xFFF\n\
                    }\n\
                    \n\
                    /// Legacy macro name for compatibility\n\
                    pub const fn VK_API_VERSION_PATCH(version: u32) -> u32 {\n\
                        vk_api_version_patch(version)\n\
                    }",
                );
            }
            // Handle null handle
            "VK_NULL_HANDLE" => {
                code.push_str("pub const VK_NULL_HANDLE: u64 = 0;\n");
            }
            _ => {
                // Try to convert to proper constant
                if let Some(constant_def) =
                    self.try_convert_to_constant(&macro_def.name, cleaned_definition)
                {
                    code.push_str(&constant_def);
                } else {
                    // Default object-like macro handling - properly comment out C code
                    let (rust_type, rust_value) = self.convert_macro_value(cleaned_definition);

                    // Only generate const if we got a valid Rust value
                    if !rust_value.contains("Complex macro") {
                        code.push_str(&format!(
                            "pub const {}: {} = {};\n",
                            macro_def.name, rust_type, rust_value
                        ));
                    } else {
                        // Comment out complex definitions
                        code.push_str(&format!(
                            "// TODO: Implement macro {} manually\n",
                            macro_def.name
                        ));

                        // Comment out each line of the original C definition
                        for line in cleaned_definition.lines() {
                            code.push_str(&format!("// {}\n", line));
                        }
                    }
                }
            }
        }

        code
    }

    /// Generate conditional compilation macro
    fn generate_conditional_macro(
        &self,
        macro_def: &MacroDefinition,
        cleaned_definition: &str,
    ) -> String {
        let mut code = String::new();

        // Documentation
        code.push_str(&format!(
            "/// Conditional compilation: {} (from line {})\n",
            macro_def.name,
            self.fmt_line(macro_def.source_line)
        ));

        // Convert to Rust cfg attribute
        let condition = self.convert_conditional(cleaned_definition);
        code.push_str(&format!("// #[cfg({})]\n", condition));
        code.push_str(&format!("// Original: {}\n", cleaned_definition));

        code
    }

    /// Generate generic macro (fallback)
    fn generate_generic_macro(
        &self,
        macro_def: &MacroDefinition,
        cleaned_definition: &str,
    ) -> String {
        let mut code = String::new();

        // Documentation
        code.push_str(&format!(
            "// Vulkan macro: {} (from line {})\n",
            macro_def.name,
            self.fmt_line(macro_def.source_line)
        ));

        // If it looks like a simple numeric constant, generate as const
        if cleaned_definition
            .chars()
            .all(|c| c.is_ascii_digit() || "xXabcdefABCDEF".contains(c))
        {
            let (rust_type, rust_value) = self.convert_macro_value(cleaned_definition);
            code.push_str(&format!(
                "pub const {}: {} = {};\n",
                macro_def.name, rust_type, rust_value
            ));
        } else {
            // Otherwise, generate a comment for manual implementation
            code.push_str(&format!(
                "// TODO: Implement macro {} manually\n",
                macro_def.name
            ));

            // Comment out each line of the original C definition
            for line in cleaned_definition.lines() {
                code.push_str(&format!("// {}\n", line));
            }
        }

        code
    }

    /// Convert macro value to appropriate Rust type and value
    fn convert_macro_value(&self, definition: &str) -> (&'static str, String) {
        let trimmed = definition.trim();

        // Hexadecimal values
        if trimmed.starts_with("0x") || trimmed.starts_with("0X") {
            return ("u32", trimmed.to_string());
        }

        // Decimal integers
        if trimmed.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(val) = trimmed.parse::<i32>() {
                if val >= 0 {
                    return ("u32", format!("{}", val));
                } else {
                    return ("i32", format!("{}", val));
                }
            }
        }

        // String literals
        if trimmed.starts_with('"') && trimmed.ends_with('"') {
            return ("&str", trimmed.to_string());
        }

        // Float values
        if trimmed.contains('.') && trimmed.parse::<f32>().is_ok() {
            return ("f32", trimmed.to_string());
        }

        // Boolean-like values
        if trimmed == "1" {
            return ("bool", "true".to_string());
        }
        if trimmed == "0" {
            return ("bool", "false".to_string());
        }

        // For complex C definitions, create a comment instead of invalid Rust
        if trimmed.contains("#define") || trimmed.contains("//") {
            return (
                "&str",
                format!(
                    "\"{}\"",
                    "// Complex macro - manual implementation required"
                ),
            );
        }

        // Default to string constant
        ("&str", format!("\"{}\"", trimmed.replace('"', "\\\"")))
    }

    /// Clean HTML entities from macro definitions
    fn clean_html_entities(&self, definition: &str) -> String {
        definition
            .replace("&gt;", ">")
            .replace("&lt;", "<")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
    }

    /// Try to convert a macro definition to a proper Rust constant.
    /// Dynamically handles VK_MAKE_API_VERSION(...) and simple numeric values
    /// so that version constants are derived from vk.xml rather than hardcoded.
    fn try_convert_to_constant(&self, name: &str, definition: &str) -> Option<String> {
        // Extract the value part from #define lines
        let value_text = if let Some(define_pos) = definition.find("#define") {
            let after_define = &definition[define_pos + 7..];
            let parts: Vec<&str> = after_define
                .trim()
                .splitn(2, |c: char| c.is_whitespace())
                .collect();
            if parts.len() >= 2 {
                // Take only the first line after the name
                parts[1]
                    .trim()
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            } else {
                return None;
            }
        } else {
            definition.trim().to_string()
        };

        // Remove trailing C comments
        let value_clean = if let Some(comment_pos) = value_text.find("//") {
            value_text[..comment_pos].trim().to_string()
        } else {
            value_text.trim().to_string()
        };

        if value_clean.is_empty() {
            return None;
        }

        // Handle VK_MAKE_API_VERSION(variant, major, minor, patch) where
        // every argument is a literal — fold it to a single integer
        // value at codegen time.
        if value_clean.contains("VK_MAKE_API_VERSION") {
            if let Some(computed) = Self::eval_make_api_version(&value_clean) {
                return Some(format!("pub const {}: u32 = {};\n", name, computed));
            }
            // Otherwise: try to translate it to a Rust const expression
            // that defers evaluation until Rust compile time. This is
            // the path VK_HEADER_VERSION_COMPLETE takes — its body is
            // `VK_MAKE_API_VERSION(0, 1, 4, VK_HEADER_VERSION)` where
            // VK_HEADER_VERSION is another const we emit, so the
            // resulting Rust expression
            //   vk_make_api_version(0, 1, 4, VK_HEADER_VERSION)
            // is evaluated by `rustc` and remains correct as the
            // spec version changes.
            if let Some(rust_expr) = Self::translate_make_api_version_invocation(&value_clean) {
                return Some(format!("pub const {}: u32 = {};\n", name, rust_expr));
            }
        }

        // Handle simple numeric definitions
        if let Ok(num) = value_clean.parse::<u64>() {
            return Some(format!("pub const {}: u32 = {};\n", name, num));
        }
        if let Ok(num) = value_clean.parse::<i64>() {
            if num < 0 {
                return Some(format!("pub const {}: i32 = {};\n", name, num));
            }
        }

        None
    }

    /// Translate a `VK_MAKE_API_VERSION(a, b, c, d)` invocation into a
    /// Rust const-fn call expression. Each argument is allowed to be
    /// either an integer literal or a Vulkan identifier (which will
    /// resolve to another `pub const` at Rust compile time).
    ///
    /// Returns `None` if the expression doesn't match the pattern or
    /// any argument fails the simple-token check (we deliberately
    /// refuse arbitrary C expressions to keep this safe).
    fn translate_make_api_version_invocation(expr: &str) -> Option<String> {
        // Find the VK_MAKE_API_VERSION call site.
        let call_start = expr.find("VK_MAKE_API_VERSION")?;
        let after_name = &expr[call_start + "VK_MAKE_API_VERSION".len()..];
        let open = after_name.find('(')?;
        // Find the matching close paren by counting depth.
        let mut depth = 0i32;
        let mut close_idx: Option<usize> = None;
        for (i, ch) in after_name[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close_idx = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close_idx?;
        let args_str = &after_name[open + 1..close];

        // Split the args naively on commas (no nested commas inside
        // expected for a 4-arg version macro).
        let args: Vec<&str> = args_str.split(',').collect();
        if args.len() != 4 {
            return None;
        }

        // Each arg must be either a decimal integer literal or a
        // C identifier (alphanumeric + underscore, leading non-digit).
        let mut rust_args: Vec<String> = Vec::new();
        for arg in &args {
            let trimmed = arg.trim();
            if trimmed.is_empty() {
                return None;
            }
            if trimmed.chars().all(|c| c.is_ascii_digit()) {
                rust_args.push(trimmed.to_string());
                continue;
            }
            // Identifier: starts with letter/underscore, rest alphanumeric+underscore.
            let mut chars = trimmed.chars();
            let first = chars.next()?;
            if !(first.is_ascii_alphabetic() || first == '_') {
                return None;
            }
            if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return None;
            }
            // Ident — emit as-is so Rust resolves it to another `pub const`.
            rust_args.push(trimmed.to_string());
        }

        Some(format!(
            "vk_make_api_version({}, {}, {}, {})",
            rust_args[0], rust_args[1], rust_args[2], rust_args[3]
        ))
    }

    /// Try to transpile a C function-like macro into a Rust const fn.
    /// Parses the C macro body from the definition, strips casts, and emits
    /// equivalent Rust using the same operators (<<, >>, |, &).
    fn try_transpile_c_macro(
        &self,
        name: &str,
        params: &[String],
        definition: &str,
        deprecated: Option<&str>,
    ) -> Option<String> {
        // Extract the macro body: everything after `#define NAME(params)` on the same line
        let body = Self::extract_c_macro_body(name, definition)?;
        let body = body.trim();
        if body.is_empty() {
            return None;
        }

        // Transpile the C expression to Rust
        let rust_body = Self::transpile_c_expr(body)?;

        // Strip outermost redundant parens: "(expr)" -> "expr" if the whole body is wrapped
        let rust_body = Self::strip_outer_parens(&rust_body);

        // Build the Rust const fn
        let snake_name = Self::to_snake_case(name);
        let param_list: String = params
            .iter()
            .map(|p| format!("{}: u32", p))
            .collect::<Vec<_>>()
            .join(", ");

        let mut code = String::new();

        if let Some(dep) = deprecated {
            if dep == "true" {
                code.push_str("#[deprecated]\n");
            } else {
                code.push_str(&format!("#[deprecated(note = \"{}\")]\n", dep));
            }
        }

        code.push_str(&format!(
            "pub const fn {}({}) -> u32 {{\n    {}\n}}\n\n",
            snake_name, param_list, rust_body
        ));

        // Also emit with original C name for compatibility
        let call_args: String = params.join(", ");
        code.push_str(&format!(
            "#[allow(non_snake_case)]\npub const fn {}({}) -> u32 {{\n    {}({})\n}}\n",
            name, param_list, snake_name, call_args
        ));

        Some(code)
    }

    /// Extract the body of a C #define macro from its full definition text.
    /// For `#define VK_MAKE_API_VERSION(variant, major, minor, patch) \
    ///     ((((uint32_t)(variant)) << 29U) | ...)`
    /// returns the expression after the parameter list.
    fn extract_c_macro_body(name: &str, definition: &str) -> Option<String> {
        // Find the #define line
        let define_idx = definition.find("#define")?;
        let after_define = &definition[define_idx + 7..];

        // Skip whitespace and the macro name
        let trimmed = after_define.trim_start();
        let after_name = trimmed.strip_prefix(name)?;

        // Find the parameter list: (param1, param2, ...)
        let paren_start = after_name.find('(')?;
        let mut depth = 0;
        let mut paren_end = None;
        for (i, ch) in after_name[paren_start..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        paren_end = Some(paren_start + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }

        let body_start = paren_end?;
        let body = &after_name[body_start..];

        // Join continuation lines (backslash-newline) and take content up to
        // end-of-definition (next unescaped newline or end of string)
        let joined = body.replace("\\\n", " ").replace("\\n", " ");
        // Take only the first logical line (stop at a line that isn't a continuation)
        let first_line = joined.lines().next().unwrap_or("");

        // Remove trailing C comments
        let without_comment = if let Some(idx) = first_line.find("//") {
            &first_line[..idx]
        } else {
            first_line
        };

        let result = without_comment.trim().to_string();
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Transpile a C expression to Rust.
    /// Strips C casts like `(uint32_t)(x)`, removes `U` suffixes from
    /// integer literals, and preserves operators (<<, >>, |, &, ~).
    fn transpile_c_expr(expr: &str) -> Option<String> {
        let mut result = expr.to_string();

        // Decode XML entities that may remain
        result = result.replace("&lt;", "<");
        result = result.replace("&gt;", ">");
        result = result.replace("&amp;", "&");

        // Remove C casts: (uint32_t), (int32_t), (uint64_t), etc.
        let cast_re =
            regex::Regex::new(r"\(\s*(?:uint32_t|int32_t|uint64_t|int64_t|uint8_t|int)\s*\)")
                .ok()?;
        result = cast_re.replace_all(&result, "").to_string();

        // Remove U/UL/ULL suffixes from numeric literals (decimal and hex)
        let suffix_re = regex::Regex::new(r"(\b(?:0[xX][0-9a-fA-F]+|\d+))[Uu][Ll]{0,2}\b").ok()?;
        result = suffix_re.replace_all(&result, "$1").to_string();

        // Replace C bitwise NOT (~) with Rust (!)
        result = result.replace('~', "!");

        // Clean up whitespace
        let ws_re = regex::Regex::new(r"\s+").ok()?;
        result = ws_re.replace_all(&result, " ").to_string();

        // Collapse redundant parentheses: ((x)) -> (x), (identifier) -> identifier
        let double_paren = regex::Regex::new(r"\(\(([^()]+)\)\)").ok()?;
        for _ in 0..5 {
            let before = result.clone();
            result = double_paren.replace_all(&result, "($1)").to_string();
            if result == before {
                break;
            }
        }
        // Remove parens around simple identifiers: (version) -> version
        let trivial_paren = regex::Regex::new(r"\(([A-Za-z_][A-Za-z0-9_]*)\)").ok()?;
        result = trivial_paren.replace_all(&result, "$1").to_string();

        let result = result.trim().to_string();

        // Validate: only allow safe characters (parens, operators, identifiers, numbers, spaces)
        if result.chars().all(|c| {
            c.is_alphanumeric()
                || c == '_'
                || c == ' '
                || c == '('
                || c == ')'
                || c == '<'
                || c == '>'
                || c == '|'
                || c == '&'
                || c == '^'
                || c == '!'
                || c == '+'
                || c == '-'
                || c == '*'
                || c == '/'
                || c == 'x'
                || c == 'X' // for hex literals like 0x3FF
        }) {
            Some(result)
        } else {
            None
        }
    }

    /// Strip outermost redundant parentheses from an expression.
    /// "(a | b)" -> "a | b", but "(a) | (b)" stays unchanged.
    fn strip_outer_parens(expr: &str) -> String {
        let s = expr.trim();
        if s.starts_with('(') && s.ends_with(')') {
            // Check that the outer parens are matched (not "(a) | (b)")
            let mut depth = 0;
            let mut outer_matched = true;
            for (i, ch) in s.char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 && i < s.len() - 1 {
                            outer_matched = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if outer_matched && depth == 0 {
                return s[1..s.len() - 1].trim().to_string();
            }
        }
        s.to_string()
    }

    /// Convert a SCREAMING_SNAKE_CASE name to snake_case
    fn to_snake_case(name: &str) -> String {
        name.to_lowercase()
    }

    /// For macros with complex preprocessor conditionals (like VK_NULL_HANDLE),
    /// scan all `#define NAME value` lines and use the simplest one (the final
    /// unconditional fallback). The value is derived from vk.xml, not hardcoded.
    fn try_extract_fallback_define(&self, name: &str, definition: &str) -> Option<String> {
        let define_prefix = format!("#define {}", name);
        let mut best_value: Option<String> = None;

        for line in definition.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix(&define_prefix) {
                let value = rest.trim();
                // Skip lines that reference other macros (nullptr, (void*)0, etc.)
                // Prefer simple numeric literals
                if value.is_empty() {
                    continue;
                }
                // Strip C suffixes
                let clean = value.trim_end_matches(['U', 'u', 'L', 'l']);
                if clean.parse::<u64>().is_ok() {
                    best_value = Some(clean.to_string());
                    // Don't break — prefer later (more unconditional) definitions
                }
            }
        }

        best_value.map(|val| format!("pub const {}: u64 = {};\n", name, val))
    }

    /// Evaluate VK_MAKE_API_VERSION(variant, major, minor, patch) at build time.
    /// All arguments must be numeric literals for this to succeed.
    fn eval_make_api_version(expr: &str) -> Option<u32> {
        let start = expr.find('(')?;
        let end = expr.rfind(')')?;
        let args_str = &expr[start + 1..end];
        let args: Vec<&str> = args_str.split(',').collect();
        if args.len() != 4 {
            return None;
        }
        let variant: u32 = args[0].trim().parse().ok()?;
        let major: u32 = args[1].trim().parse().ok()?;
        let minor: u32 = args[2].trim().parse().ok()?;
        let patch: u32 = args[3].trim().parse().ok()?;
        Some((variant << 29) | (major << 22) | (minor << 12) | patch)
    }

    /// Convert conditional macro to Rust cfg condition
    fn convert_conditional(&self, definition: &str) -> String {
        let trimmed = definition.trim();

        // Common patterns
        if trimmed.contains("defined(") {
            // Extract the defined symbol
            if let Some(start) = trimmed.find("defined(") {
                if let Some(end) = trimmed[start..].find(')') {
                    let symbol = &trimmed[start + 8..start + end];
                    return format!("feature = \"{}\"", symbol.to_lowercase());
                }
            }
        }

        // Platform checks
        if trimmed.contains("WIN32") || trimmed.contains("_WIN32") {
            return "target_os = \"windows\"".to_string();
        }
        if trimmed.contains("__linux__") {
            return "target_os = \"linux\"".to_string();
        }
        if trimmed.contains("__APPLE__") {
            return "target_os = \"macos\"".to_string();
        }
        if trimmed.contains("__ANDROID__") {
            return "target_os = \"android\"".to_string();
        }

        // Default
        format!("feature = \"{}\"", trimmed.to_lowercase())
    }
}

impl GeneratorModule for MacroGenerator {
    fn name(&self) -> &str {
        "MacroGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["macros.json".to_string()]
    }

    fn output_file(&self) -> String {
        "macros.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        Vec::new() // Macros typically don't depend on other modules
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("macros.json");
        let content =
            fs::read_to_string(&input_path).map_err(|_| GeneratorError::MissingInput {
                path: input_path.display().to_string(),
            })?;

        let macros: Vec<MacroDefinition> = serde_json::from_str(&content)?;

        // Generate code
        let mut code = String::new();

        // Add file header
        code.push_str("// === MACRO DEFINITIONS (FIXED) ===\n\n");

        // Track generated macros to avoid duplicates
        let mut generated_macros = std::collections::HashSet::new();

        // Add necessary imports for complex functions only if not already in macros
        let has_make_api_version = macros
            .iter()
            .any(|m| m.name == "VK_MAKE_API_VERSION" || m.name == "vk_make_api_version");

        if !has_make_api_version {
            code.push_str("// Required function definitions for version macros\n");
            code.push_str("pub const fn vk_make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {\n");
            code.push_str("    (variant << 29) | (major << 22) | (minor << 12) | patch\n");
            code.push_str("}\n\n");
            generated_macros.insert("vk_make_api_version".to_string());
            generated_macros.insert("VK_MAKE_API_VERSION".to_string());
        }

        // Generate individual macros
        for macro_def in &macros {
            if !generated_macros.contains(&macro_def.name) {
                code.push_str(&self.generate_macro(macro_def));
                generated_macros.insert(macro_def.name.clone());
            }
        }

        // Write output file
        let output_path = output_dir.join(self.output_file());
        fs::write(output_path, code)?;

        Ok(())
    }

    fn metadata(&self) -> GeneratorMetadata {
        GeneratorMetadata {
            defined_types: vec![], // Macros don't typically define types
            used_types: vec!["u32".to_string(), "i32".to_string(), "f32".to_string()],
            has_forward_declarations: false,
            priority: 10, // Early, but after basic types
        }
    }
}

impl MacroGenerator {
    /// Format optional source_line for display in generated comments
    fn fmt_line(&self, line: Option<usize>) -> String {
        match line {
            Some(n) => n.to_string(),
            None => "?".to_string(),
        }
    }
}
