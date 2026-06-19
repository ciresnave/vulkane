//! Enum generator module
//!
//! Generates Rust enums from enums.json intermediate file

use crate::codegen::logging::{log_debug, log_info};
use crate::parser::vk_types::{ConstantDefinition, EnumDefinition, EnumValue};
use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorMetadata, GeneratorModule, GeneratorResult};

/// Generator module for Vulkan enums
pub struct EnumGenerator;

impl Default for EnumGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl EnumGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Parse enum value to integer for conflict detection
    fn parse_enum_value(&self, value: &str) -> Option<i64> {
        // A more robust parser that can evaluate simple integer expressions
        // commonly found in vk.xml: decimal, hex, negative numbers,
        // parenthesized expressions, shifts (<<, >>), and bitwise ops (|, &, ^),
        // and unary ~ operator.
        // We'll implement a small recursive-descent evaluator that returns
        // a signed 128-bit integer when successful, otherwise None.

        // Helper: tokenize the input into numbers, operators, and parens
        #[derive(Debug, Clone)]
        enum Tok {
            Num(i128),
            Op(char),
            Shift(String),
            Caret,
            Eof,
            LParen,
            RParen,
        }

        fn tokenize(s: &str) -> Vec<Tok> {
            let mut out = Vec::new();
            let mut chars = s.trim().chars().peekable();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() {
                    chars.next();
                    continue;
                }
                if c == '(' {
                    out.push(Tok::LParen);
                    chars.next();
                    continue;
                }
                if c == ')' {
                    out.push(Tok::RParen);
                    chars.next();
                    continue;
                }
                // shifts
                if c == '<' || c == '>' {
                    let mut s = String::new();
                    s.push(c);
                    chars.next();
                    if let Some(&next) = chars.peek() {
                        if next == c {
                            s.push(next);
                            chars.next();
                        }
                    }
                    out.push(Tok::Shift(s));
                    continue;
                }
                if c == '^' {
                    out.push(Tok::Caret);
                    chars.next();
                    continue;
                }
                if c == '~' || c == '&' || c == '|' || c == '+' || c == '-' || c == '*' || c == '/'
                {
                    out.push(Tok::Op(c));
                    chars.next();
                    continue;
                }

                // number (decimal or hex)
                if c.is_ascii_digit() || (c == '0') {
                    let mut buf = String::new();
                    // support hex 0x...
                    if c == '0' {
                        buf.push(c);
                        chars.next();
                        if let Some(&nx) = chars.peek() {
                            if nx == 'x' || nx == 'X' {
                                buf.push(nx);
                                chars.next();
                                // collect hex digits
                                while let Some(&h) = chars.peek() {
                                    if h.is_ascii_hexdigit() {
                                        buf.push(h);
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                                // parse hex
                                if let Ok(v) = i128::from_str_radix(&buf[2..], 16) {
                                    out.push(Tok::Num(v));
                                    continue;
                                }
                            }
                        }
                        // if not hex, fallthrough to parse as decimal sequence starting with '0'
                    }

                    // decimal sequence (allow trailing U/UL suffixes)
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() {
                            buf.push(d);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    // skip common suffixes (U, L, etc.)
                    while let Some(&sfx) = chars.peek() {
                        if sfx == 'U' || sfx == 'u' || sfx == 'L' || sfx == 'l' {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if !buf.is_empty() {
                        if let Ok(v) = buf.parse::<i128>() {
                            out.push(Tok::Num(v));
                            continue;
                        }
                    }
                }

                // Unknown token: consume and bail
                chars.next();
                return vec![];
            }
            out.push(Tok::Eof);
            out
        }

        // Recursive descent parser over token list
        fn parse(tokens: &[Tok]) -> Option<(i128, usize)> {
            // implement precedence climbing
            // We'll walk with an index
            fn parse_primary(toks: &[Tok], idx: &mut usize) -> Option<i128> {
                match toks.get(*idx)? {
                    Tok::Num(n) => {
                        *idx += 1;
                        Some(*n)
                    }
                    Tok::LParen => {
                        *idx += 1;
                        let v = parse_expr(toks, idx)?;
                        // expect RParen
                        if let Some(Tok::RParen) = toks.get(*idx) {
                            *idx += 1;
                            Some(v)
                        } else {
                            None
                        }
                    }
                    Tok::Op(op) if *op == '-' || *op == '+' || *op == '~' => {
                        let c = *op;
                        *idx += 1;
                        let rhs = parse_primary(toks, idx)?;
                        match c {
                            '-' => Some(-rhs),
                            '+' => Some(rhs),
                            '~' => Some(!rhs),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }

            fn parse_shift(toks: &[Tok], idx: &mut usize) -> Option<i128> {
                let mut lhs = parse_primary(toks, idx)?;
                loop {
                    match toks.get(*idx) {
                        Some(Tok::Shift(s)) if s == "<<" => {
                            *idx += 1;
                            let rhs = parse_primary(toks, idx)?;
                            lhs <<= rhs;
                        }
                        Some(Tok::Shift(s)) if s == ">>" => {
                            *idx += 1;
                            let rhs = parse_primary(toks, idx)?;
                            lhs >>= rhs;
                        }
                        _ => break,
                    }
                }
                Some(lhs)
            }

            fn parse_bit_and(toks: &[Tok], idx: &mut usize) -> Option<i128> {
                let mut lhs = parse_shift(toks, idx)?;
                while let Some(Tok::Op('&')) = toks.get(*idx) {
                    *idx += 1;
                    let rhs = parse_shift(toks, idx)?;
                    lhs &= rhs;
                }
                Some(lhs)
            }

            fn parse_bit_xor(toks: &[Tok], idx: &mut usize) -> Option<i128> {
                let mut lhs = parse_bit_and(toks, idx)?;
                while let Some(Tok::Caret) = toks.get(*idx) {
                    *idx += 1;
                    let rhs = parse_bit_and(toks, idx)?;
                    lhs ^= rhs;
                }
                Some(lhs)
            }

            fn parse_bit_or(toks: &[Tok], idx: &mut usize) -> Option<i128> {
                let mut lhs = parse_bit_xor(toks, idx)?;
                while let Some(Tok::Op('|')) = toks.get(*idx) {
                    *idx += 1;
                    let rhs = parse_bit_xor(toks, idx)?;
                    lhs |= rhs;
                }
                Some(lhs)
            }

            fn parse_expr(toks: &[Tok], idx: &mut usize) -> Option<i128> {
                // currently only bitwise and shifts are common; allow add/sub too
                let mut lhs = parse_bit_or(toks, idx)?;
                loop {
                    match toks.get(*idx) {
                        Some(Tok::Op('+')) => {
                            *idx += 1;
                            let rhs = parse_bit_or(toks, idx)?;
                            lhs += rhs;
                        }
                        Some(Tok::Op('-')) => {
                            *idx += 1;
                            let rhs = parse_bit_or(toks, idx)?;
                            lhs -= rhs;
                        }
                        _ => break,
                    }
                }
                Some(lhs)
            }

            let mut idx = 0usize;
            let res = parse_expr(tokens, &mut idx)?;
            Some((res, idx))
        }

        // Preprocess the input a bit: remove trailing C suffixes and surrounding parens
        let mut s = value.trim().to_string();
        // Trim common unsigned/long suffixes (U, UL, ULL, L)
        while s.ends_with('U') || s.ends_with('u') || s.ends_with('L') || s.ends_with('l') {
            s.pop();
            s = s.trim().to_string();
        }

        // Tokenize
        let toks = tokenize(&s);
        if toks.is_empty() {
            return None;
        }

        if let Some((val, _used)) = parse(&toks) {
            // Attempt to fit into i64
            if val <= i64::MAX as i128 && val >= i64::MIN as i128 {
                return Some(val as i64);
            }
        }

        None
    }

    /// Generate Rust code for a single enum
    fn generate_enum(
        &self,
        enum_def: &EnumDefinition,
        constants_present: &std::collections::HashSet<String>,
    ) -> String {
        let mut code = String::new();

        // Documentation comment from vk.xml if available, otherwise the type name
        if let Some(comment) = &enum_def.comment {
            for line in comment.lines() {
                code.push_str(&format!(
                    "/// {}\n",
                    crate::codegen::sanitize_doc_line(line)
                ));
            }
        } else {
            code.push_str(&format!("/// Vulkan enum: `{}`\n", enum_def.name));
        }

        // Derive attributes with Default if needed
        code.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]\n");
        code.push_str("#[repr(i32)]\n");

        // If this enum_def represents a bag of constants (API Constants),
        // emit them as `pub const` definitions instead of a Rust enum. This
        // avoids mixed-type discriminants (floats, unsigned, etc.) which
        // are invalid as enum variants.
        if enum_def.enum_type == "constants" {
            for val in &enum_def.values {
                // Skip duplicates that are already emitted by constants.json
                if constants_present.contains(&val.name) {
                    continue;
                }

                // Documentation comment
                if let Some(line) = val.source_line {
                    code.push_str(&format!("/// {} (from line {})\n", val.name, line));
                } else {
                    code.push_str(&format!("/// {}\n", val.name));
                }

                let raw_val = val.value.as_deref().unwrap_or("0");
                let rust_type = infer_const_type(raw_val);
                let rust_value = map_const_value(raw_val, &rust_type);

                code.push_str(&format!(
                    "pub const {}: {} = {};\n\n",
                    val.name, rust_type, rust_value
                ));
            }

            return code;
        }

        // Sanitize enum name into a valid Rust identifier (no spaces or punctuation)
        let enum_name = enum_def
            .name
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>()
            .trim_matches('_')
            .to_string();

        // Skip empty enums - they can't have #[repr]
        if enum_def.values.is_empty() {
            code.clear();
            code.push_str(&format!("pub type {} = i32;\n\n", enum_name));
            return code;
        }

        // Check if any value exceeds i32 range (bitmask enums with large bit positions)
        let has_large_values = enum_def.values.iter().any(|v| {
            if let Some(bp) = &v.bitpos {
                bp.parse::<u32>().unwrap_or(0) >= 31
            } else if let Some(val) = &v.value {
                val.parse::<i64>().unwrap_or(0).unsigned_abs() > i32::MAX as u64
            } else {
                false
            }
        });

        // Bitmask enums or enums with large values: emit as pub const instead of Rust enum
        if enum_def.enum_type == "bitmask" || has_large_values {
            code.clear();
            let base_type = if has_large_values || enum_def.bitwidth.as_deref() == Some("64") {
                "u64"
            } else {
                "u32"
            };
            code.push_str(&format!("pub type {} = {};\n", enum_name, base_type));
            for value in &enum_def.values {
                let val_str = if let Some(bp) = &value.bitpos {
                    let bp_val: u64 = bp.parse().unwrap_or(0);
                    format!("1{} << {}", base_type, bp_val)
                } else if let Some(v) = &value.value {
                    self.format_enum_value(v)
                } else if let Some(alias) = &value.alias {
                    self.format_enum_value_name(alias)
                } else {
                    "0".to_string()
                };
                code.push_str(&format!(
                    "pub const {}: {} = {};\n",
                    self.format_enum_value_name(&value.name),
                    enum_name,
                    val_str
                ));
            }
            code.push('\n');
            return code;
        }

        // Start enum definition
        code.push_str(&format!("pub enum {} {{\n", enum_name));

        // Track used values to detect duplicates.
        // Duplicates (promoted extensions with same value) become pub const aliases.
        let mut used_values = std::collections::HashMap::<i64, String>::new();
        let mut alias_values: Vec<(String, String, String)> = Vec::new(); // (alias_name, original_variant, value_str)

        for value in &enum_def.values {
            // Skip alias-only entries (they reference another value by name)
            if value.is_alias {
                if let Some(alias_target) = &value.alias {
                    alias_values.push((
                        self.format_enum_value_name(&value.name),
                        format!(
                            "{}::{}",
                            enum_name,
                            self.format_enum_value_name(alias_target)
                        ),
                        String::new(),
                    ));
                }
                continue;
            }

            // Prefer the explicit value, fall back to bitpos.
            // The actual `1 << bitpos` computation happens below.
            let value_str = value
                .value
                .as_deref()
                .or(value.bitpos.as_deref())
                .unwrap_or("0");

            // For bitpos values, compute the actual value
            let formatted_value = if value.bitpos.is_some() && value.value.is_none() {
                let bp: i64 = value_str.parse().unwrap_or(0);
                format!("{}", 1i64 << bp)
            } else {
                self.format_enum_value(value_str)
            };

            let parsed = self.parse_enum_value(&formatted_value);

            // Emit doc comment if present
            let emit_doc = |code: &mut String| {
                if let Some(c) = &value.comment {
                    for line in c.lines() {
                        code.push_str(&format!(
                            "    /// {}\n",
                            crate::codegen::sanitize_doc_line(line)
                        ));
                    }
                }
            };

            if let Some(numeric_val) = parsed {
                if let Some(existing_name) = used_values.get(&numeric_val) {
                    // Duplicate value - emit as a const alias instead
                    alias_values.push((
                        self.format_enum_value_name(&value.name),
                        format!("{}::{}", enum_name, existing_name),
                        formatted_value,
                    ));
                    continue;
                }
                let variant_name = self.format_enum_value_name(&value.name);
                used_values.insert(numeric_val, variant_name.clone());
                emit_doc(&mut code);
                code.push_str(&format!("    {} = {},\n", variant_name, formatted_value));
            } else {
                let variant_name = self.format_enum_value_name(&value.name);
                emit_doc(&mut code);
                code.push_str(&format!("    {} = {},\n", variant_name, formatted_value));
            }
        }

        code.push_str("}\n\n");

        // Emit duplicate values and aliases as pub const.
        // Build a map from numeric value -> first variant name for resolution.
        let mut value_to_variant: std::collections::HashMap<i64, String> =
            std::collections::HashMap::new();
        let mut variant_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        {
            let mut uv = std::collections::HashSet::<i64>::new();
            for value in &enum_def.values {
                if value.is_alias {
                    continue;
                }
                let value_str = value.value.as_deref().unwrap_or("0");
                let formatted = if value.bitpos.is_some() && value.value.is_none() {
                    let bp: i64 = value.bitpos.as_deref().unwrap_or("0").parse().unwrap_or(0);
                    format!("{}", 1i64 << bp)
                } else {
                    self.format_enum_value(value_str)
                };
                if let Some(n) = self.parse_enum_value(&formatted) {
                    if uv.insert(n) {
                        let vn = self.format_enum_value_name(&value.name);
                        value_to_variant.insert(n, vn.clone());
                        variant_set.insert(vn);
                    }
                }
            }
        }

        for (alias_name, target, value_str) in &alias_values {
            // Resolve: if we have a numeric value, find the real variant
            // Otherwise, resolve the alias target through the variant set
            let resolved = if !value_str.is_empty() {
                if let Some(n) = self.parse_enum_value(value_str) {
                    value_to_variant
                        .get(&n)
                        .map(|v| format!("{}::{}", enum_name, v))
                } else {
                    None
                }
            } else if let Some(after) = target.strip_prefix(&format!("{}::", enum_name)) {
                if variant_set.contains(after) {
                    Some(target.clone())
                } else {
                    None // Can't resolve
                }
            } else {
                None
            };

            if let Some(resolved_target) = resolved {
                code.push_str(&format!(
                    "pub const {}: {} = {};\n",
                    alias_name, enum_name, resolved_target
                ));
            }
        }
        code.push('\n');

        // Generate Default implementation - be more aggressive about creating defaults
        code.push_str(&self.generate_enum_default_impl(enum_def));

        // Generate implementation block
        code.push_str(&self.generate_enum_impl(enum_def));

        code
    }

    /// Format enum value name (remove enum prefix if present)
    fn format_enum_value_name(&self, name: &str) -> String {
        // Common Vulkan enum prefixes to remove
        let prefixes = ["VK_", "VkResult", "VkFormat", "VkImageType"];

        for prefix in &prefixes {
            if let Some(without_prefix) = name.strip_prefix(prefix) {
                return without_prefix
                    .strip_prefix('_')
                    .unwrap_or(without_prefix)
                    .to_string();
            }
        }

        name.to_string()
    }

    /// Sanitize enum type name to valid Rust identifier (same logic as used when emitting)
    fn sanitize_enum_name(&self, name: &str) -> String {
        name.chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>()
            .trim_matches('_')
            .to_string()
    }

    /// Format enum value (handle different value formats)
    fn format_enum_value(&self, value: &str) -> String {
        let orig = value.trim();
        // If this looks like a shift/bitwise expression, preserve the original
        // formatting including parentheses so tests expecting e.g. "(1 << 0)" pass.
        if orig.contains("<<") || orig.contains("|") || orig.contains("&") {
            return orig.to_string();
        }

        let mut v = orig.to_string();

        // Strip unsigned/long/float suffixes common in C (U, UL, ULL, F)
        v = v
            .trim_end_matches(['U', 'u', 'L', 'l', 'F', 'f'])
            .to_string();

        // Convert bitwise not of unsigned (~0U) into Rust literal using !0u32 or !0u64
        if v == "~0" {
            return "!0u32".to_string();
        }
        if v == "~0U" || v == "~0u" {
            return "!0u32".to_string();
        }
        if v == "~0ULL" || v.eq_ignore_ascii_case("~0ull") {
            return "!0u64".to_string();
        }

        // Handle hex values
        if v.starts_with("0x") || v.starts_with("0X") {
            return v;
        }

        // Handle negative values
        if v.starts_with('-') {
            return v;
        }

        // Handle bit operations like 1 << 0 (we already returned if original
        // had parentheses or other operators). If this expression now contains
        // shift operators, return as-is.
        if v.contains("<<") || v.contains("|") || v.contains("&") {
            return v;
        }

        // Default: return as-is
        v
    }

    /// Generate Default implementation for enum (uses first zero-valued variant or first variant)
    fn generate_enum_default_impl(&self, enum_def: &EnumDefinition) -> String {
        let mut code = String::new();

        // Only consider non-alias values that are actual enum variants
        let non_alias_values: Vec<_> = enum_def.values.iter().filter(|v| !v.is_alias).collect();

        // Find the first variant with value 0, or just use the first non-alias variant
        let default_variant = non_alias_values
            .iter()
            .find(|v| v.value.as_deref() == Some("0"))
            .or_else(|| non_alias_values.first());

        if let Some(variant) = default_variant {
            let enum_name = self.sanitize_enum_name(&enum_def.name);
            code.push_str(&format!("impl Default for {} {{\n", enum_name));
            code.push_str("    fn default() -> Self {\n");
            code.push_str(&format!(
                "        Self::{}\n",
                self.format_enum_value_name(&variant.name)
            ));
            code.push_str("    }\n");
            code.push_str("}\n\n");
        }

        code
    }

    /// Generate implementation block for enum.
    /// Only references actual enum variants (not aliases moved to pub const).
    fn generate_enum_impl(&self, enum_def: &EnumDefinition) -> String {
        let mut code = String::new();

        let enum_name = self.sanitize_enum_name(&enum_def.name);

        // Build a set of actual variant names (non-alias, non-duplicate-value)
        // matching the same logic used in the enum definition generation.
        let mut actual_variants: Vec<(&EnumValue, String)> = Vec::new();
        let mut used_values = std::collections::HashSet::<i64>::new();
        for value in &enum_def.values {
            if value.is_alias {
                continue;
            }
            let value_str = value.value.as_deref().unwrap_or("0");
            let formatted = if value.bitpos.is_some() && value.value.is_none() {
                let bp: i64 = value.bitpos.as_deref().unwrap_or("0").parse().unwrap_or(0);
                format!("{}", 1i64 << bp)
            } else {
                self.format_enum_value(value_str)
            };
            let parsed = self.parse_enum_value(&formatted);
            if let Some(n) = parsed {
                if used_values.contains(&n) {
                    continue; // Duplicate value - was emitted as pub const alias
                }
                used_values.insert(n);
            }
            actual_variants.push((value, formatted));
        }

        code.push_str(&format!("impl {} {{\n", enum_name));

        // from_raw
        code.push_str("    #[allow(unreachable_patterns)]\n");
        code.push_str("    pub fn from_raw(value: i32) -> Option<Self> {\n");
        code.push_str("        match value {\n");
        for (val, formatted) in &actual_variants {
            code.push_str(&format!(
                "            {} => Some(Self::{}),\n",
                formatted,
                self.format_enum_value_name(&val.name)
            ));
        }
        code.push_str("            _ => None,\n");
        code.push_str("        }\n");
        code.push_str("    }\n");

        // to_raw
        code.push_str("\n    pub fn to_raw(&self) -> i32 {\n");
        code.push_str("        *self as i32\n");
        code.push_str("    }\n");

        code.push_str("}\n\n");

        // Display — allow unreachable_patterns because repr(i32) enums can hold
        // raw values not covered by the defined variants (e.g. from newer extensions)
        code.push_str(&format!("impl std::fmt::Display for {} {{\n", enum_name));
        code.push_str("    #[allow(unreachable_patterns)]\n");
        code.push_str("    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {\n");
        code.push_str("        match self {\n");
        for (val, _) in &actual_variants {
            let variant_name = self.format_enum_value_name(&val.name);
            code.push_str(&format!(
                "            Self::{} => write!(f, \"{}\"),\n",
                variant_name, val.name
            ));
        }
        code.push_str("            _ => write!(f, \"Unknown({})\", *self as i32),\n");

        code.push_str("        }\n");
        code.push_str("    }\n");
        code.push_str("}\n\n");

        code
    }
}

/// Infer a Rust type for a constant value (simple heuristic)
fn infer_const_type(value: &str) -> String {
    let v = value.trim();

    // Strip surrounding parentheses for analysis
    let mut cleaned = v;
    if cleaned.starts_with('(') && cleaned.ends_with(')') {
        cleaned = &cleaned[1..cleaned.len() - 1];
    }

    // Normalize common C suffixes for easier detection
    let cleaned = cleaned
        .trim_end_matches(['U', 'u', 'L', 'l', 'F', 'f'])
        .trim();

    // Bitwise-not patterns like ~0 or ~0U should be unsigned
    if v.starts_with('~') {
        // Choose 64-bit if ULL appears, otherwise 32-bit
        if v.to_uppercase().contains("ULL") {
            return "u64".to_string();
        }
        return "u32".to_string();
    }

    // Hex values -> unsigned (use u64 only if too large detection later)
    if cleaned.starts_with("0x") || cleaned.starts_with("0X") {
        return "u32".to_string();
    }

    // Floating point detection
    if cleaned.contains('.') || v.to_uppercase().contains('F') {
        return "f32".to_string();
    }

    // Shift or bit operations should be treated as integers
    if cleaned.contains("<<") || cleaned.contains("|") || cleaned.contains("&") {
        return "u32".to_string();
    }

    // Signed/unsigned integer detection
    if cleaned.parse::<i64>().is_ok() {
        if cleaned.starts_with('-') {
            return "i64".to_string();
        }
        return "u64".to_string();
    }

    // Fallback: keep as string only if it truly cannot be interpreted
    "&'static str".to_string()
}

/// Map C-style constant values to Rust-friendly literals
fn map_const_value(value: &str, value_type: &str) -> String {
    let mut rust_value = value.trim().to_string();

    // Strip outer parentheses
    if rust_value.starts_with('(') && rust_value.ends_with(')') {
        rust_value = rust_value[1..rust_value.len() - 1].to_string();
    }

    // Handle bitwise-not (~) mapping: prefer using explicit typed !0u32/!0u64 or std::u64::MAX
    if rust_value.starts_with('~') {
        // If ULL present, map to u64::MAX
        if rust_value.to_uppercase().contains("ULL") || rust_value.to_uppercase().contains("ULL") {
            return "u64::MAX".to_string();
        }

        // If contains explicit U suffix -> !0u32
        if rust_value.to_uppercase().contains('U') {
            return "!0u32".to_string();
        }

        // Default to !0u32
        return "!0u32".to_string();
    }

    // Remove trailing C suffixes
    rust_value = rust_value
        .trim_end_matches(['U', 'u', 'L', 'l', 'F', 'f'])
        .to_string();

    // Hex values: keep as-is
    if rust_value.starts_with("0x") || rust_value.starts_with("0X") {
        return rust_value;
    }

    // Shift/bit ops: keep as-is but ensure types align (we'll leave as original expression)
    if rust_value.contains("<<") || rust_value.contains("|") || rust_value.contains("&") {
        return rust_value;
    }

    // Numeric literal with optional sign
    if rust_value.parse::<i128>().is_ok() {
        // Ensure integer width matches inferred type
        match value_type {
            "u32" => return format!("{}u32", rust_value),
            "u64" => return format!("{}u64", rust_value),
            "i64" => return format!("{}i64", rust_value),
            "i32" => return format!("{}i32", rust_value),
            _ => return rust_value,
        }
    }

    // Floating point
    if value_type.starts_with('f') {
        // Ensure decimal point
        if !rust_value.contains('.') {
            rust_value.push_str(".0");
        }
        return rust_value;
    }

    // Fallback: return quoted string for &str
    if value_type == "&'static str" {
        return format!("\"{}\"", rust_value);
    }

    rust_value
}

impl GeneratorModule for EnumGenerator {
    fn name(&self) -> &str {
        "EnumGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["enums.json".to_string()]
    }

    fn output_file(&self) -> String {
        "enums.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        Vec::new() // Enums don't depend on other modules
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("enums.json");
        let input_content = fs::read_to_string(&input_path).map_err(GeneratorError::Io)?;

        // Parse JSON - try plain array first, then object-with-array { "enums": [...] }, then fallback to JSONL
        let enums_array: Vec<EnumDefinition> =
            match serde_json::from_str::<Vec<EnumDefinition>>(&input_content) {
                Ok(v) => v,
                Err(_) => {
                    #[derive(serde::Deserialize)]
                    struct EnumsFile {
                        enums: Vec<EnumDefinition>,
                    }

                    if let Ok(wrapper) = serde_json::from_str::<EnumsFile>(&input_content) {
                        wrapper.enums
                    } else {
                        // Fallback: parse as JSONL (one object per line)
                        let mut items = Vec::new();
                        for line in input_content.lines() {
                            if !line.trim().is_empty() {
                                if let Ok(e) = serde_json::from_str::<EnumDefinition>(line) {
                                    items.push(e);
                                }
                            }
                        }
                        items
                    }
                }
            };

        // Load constants.json (if present) to avoid emitting duplicate constants
        let mut constants_present: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let consts_path = input_dir.join("constants.json");
        if consts_path.exists() {
            if let Ok(consts_content) = fs::read_to_string(&consts_path) {
                if let Ok(consts_array) =
                    serde_json::from_str::<Vec<ConstantDefinition>>(&consts_content)
                {
                    for c in consts_array {
                        constants_present.insert(c.name);
                    }
                }
            }
        }

        // Generate code
        let mut generated_code = String::new();

        // Generate enums
        for enum_def in &enums_array {
            generated_code.push_str(&self.generate_enum(enum_def, &constants_present));
            generated_code.push('\n');
        }

        // Write output file
        let output_path = output_dir.join(self.output_file());
        fs::write(output_path, generated_code).map_err(GeneratorError::Io)?;

        // Collect and report metadata
        let _metadata = self.collect_metadata(input_dir)?;
        // Log message already handled in collect_metadata

        log_info(&format!("Generated {} enums", enums_array.len()));
        Ok(())
    }

    fn metadata(&self) -> GeneratorMetadata {
        // In a real implementation, this would extract information from the loaded enums
        // For now, we'll return placeholder data that will be populated during generate()
        GeneratorMetadata {
            defined_types: Vec::new(),
            used_types: Vec::new(),
            has_forward_declarations: false,
            priority: 20, // Enums should be generated before structs that use them
        }
    }

    /// Populate metadata from parsed enums
    fn collect_metadata(&self, input_dir: &Path) -> GeneratorResult<GeneratorMetadata> {
        let mut defined_types = Vec::new();

        // Read the enums input file
        let input_path = input_dir.join("enums.json");
        let input_content = std::fs::read_to_string(input_path).map_err(GeneratorError::Io)?;

        // Parse JSON - try plain array, then object-with-array wrapper
        let enums_array: Vec<EnumDefinition> =
            match serde_json::from_str::<Vec<EnumDefinition>>(&input_content) {
                Ok(v) => v,
                Err(_) => {
                    #[derive(serde::Deserialize)]
                    struct EnumsFile {
                        enums: Vec<EnumDefinition>,
                    }

                    let wrapper: EnumsFile =
                        serde_json::from_str(&input_content).map_err(GeneratorError::Json)?;
                    wrapper.enums
                }
            };

        // Collect all defined enum types
        for enum_def in &enums_array {
            defined_types.push(enum_def.name.clone());
        }

        // For reporting purposes
        log_debug(&format!(
            "EnumGenerator defined {} types",
            defined_types.len()
        ));

        // Enums typically don't depend on other types, so used_types is empty
        Ok(GeneratorMetadata {
            defined_types,
            used_types: Vec::new(),
            has_forward_declarations: false,
            priority: 20,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum_value_name_formatting() {
        let generator = EnumGenerator::new();

        assert_eq!(generator.format_enum_value_name("VK_SUCCESS"), "SUCCESS");
        assert_eq!(
            generator.format_enum_value_name("VK_ERROR_OUT_OF_HOST_MEMORY"),
            "ERROR_OUT_OF_HOST_MEMORY"
        );
        assert_eq!(
            generator.format_enum_value_name("CUSTOM_VALUE"),
            "CUSTOM_VALUE"
        );
    }

    #[test]
    fn test_enum_value_formatting() {
        let generator = EnumGenerator::new();

        assert_eq!(generator.format_enum_value("0"), "0");
        assert_eq!(generator.format_enum_value("0x1000"), "0x1000");
        assert_eq!(generator.format_enum_value("-1"), "-1");
        assert_eq!(generator.format_enum_value("(1 << 0)"), "(1 << 0)");
    }
}
