//! Generator for the safe-layer `DeviceFeatures` builder.
//!
//! Emits one `with_<snake_feature_name>()` method per unique boolean
//! feature bit across every Vulkan struct that extends
//! `VkPhysicalDeviceFeatures2`. Writes the output as a standalone
//! `device_features_generated.rs` file which is `include!`'d from
//! `vulkane/src/safe/features.rs`.
//!
//! # Collision resolution
//!
//! A given feature bit often appears in multiple structs — e.g.
//! `timelineSemaphore` lives in both `VkPhysicalDeviceVulkan12Features`
//! (the 1.2 aggregate) and `VkPhysicalDeviceTimelineSemaphoreFeatures`
//! (the promoted ex-extension). We emit exactly one `with_…` method
//! per snake-cased name, targeting the struct with the highest
//! "priority":
//!
//! 1. `VkPhysicalDeviceFeatures` — Vulkan 1.0 core, highest priority.
//! 2. `VkPhysicalDeviceVulkanXxFeatures` aggregates — one per core
//!    version, ordered newest-first.
//! 3. Promoted-to-core `*Features` structs with no vendor suffix.
//! 4. `*FeaturesKHR` / `*FeaturesEXT` / `*Features<Vendor>` extension
//!    structs.
//!
//! Users who need the older-extension path on a driver too old for the
//! aggregate can always chain the lower-priority struct manually via
//! `chain_extension_feature(...)`.

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorResult};
use crate::codegen::camel_to_snake;
use crate::parser::vk_types::{EnumDefinition, StructDefinition, StructMember};

/// Reserved Rust keywords that need `r#` prefixing when used as method
/// or field names. Kept in sync with the escape list in `StructGenerator`.
const RESERVED_KEYWORDS: &[&str] = &[
    "type", "match", "impl", "fn", "let", "mut", "const", "static", "if", "else", "while", "for",
    "loop", "break", "continue", "return", "struct", "enum", "trait", "mod", "pub", "use",
    "extern", "crate", "self", "Self", "super", "where", "async", "await", "dyn", "abstract",
    "become", "box", "do", "final", "macro", "override", "priv", "typeof", "unsized", "virtual",
    "yield", "try", "union", "ref",
];

fn escape_keyword(name: &str) -> String {
    if RESERVED_KEYWORDS.contains(&name) {
        format!("r#{}", name)
    } else {
        name.to_string()
    }
}

/// Assign a priority score to a feature-struct name. Higher wins when
/// the same feature bit appears in multiple structs.
fn struct_priority(name: &str) -> i32 {
    // Exact match for Vulkan 1.0 core features.
    if name == "VkPhysicalDeviceFeatures" {
        return 10_000;
    }
    // Core aggregate structs `VkPhysicalDeviceVulkanXxFeatures`.
    if let Some(tail) = name.strip_prefix("VkPhysicalDeviceVulkan")
        && let Some(ver_end) = tail.find("Features")
    {
        let ver_part = &tail[..ver_end];
        if let Ok(ver) = ver_part.parse::<i32>() {
            // Higher version = higher priority, still below 1.0 core
            // so core-1.0 bits never get routed through an aggregate.
            return 9_000 + ver;
        }
    }
    // Promoted-to-core extension struct (no vendor suffix):
    // `VkPhysicalDeviceXxFeatures`.
    if let Some(stripped) = name.strip_prefix("VkPhysicalDevice")
        && stripped.ends_with("Features")
    {
        return 5_000;
    }
    // Vendor-suffixed extension structs rank by vendor — arbitrary
    // ordering within the bucket is fine because the *name* of the
    // extension struct never matters once we've chosen the method.
    for (suffix, score) in &[
        ("FeaturesKHR", 400),
        ("FeaturesEXT", 350),
        ("FeaturesNV", 300),
        ("FeaturesAMD", 300),
        ("FeaturesARM", 300),
        ("FeaturesQCOM", 300),
        ("FeaturesINTEL", 300),
        ("FeaturesIMG", 300),
        ("FeaturesVALVE", 300),
        ("FeaturesGOOGLE", 300),
        ("FeaturesHUAWEI", 300),
        ("FeaturesMESA", 300),
    ] {
        if name.ends_with(suffix) {
            return *score;
        }
    }
    // Anything else we didn't classify — lowest priority, but still
    // emittable so the extension struct gets at least one entry point.
    0
}

/// `true` if a struct is in-scope for `DeviceFeatures` method emission
/// (extends `VkPhysicalDeviceFeatures2` OR is the 1.0 core features
/// struct itself). Structs whose sType enumerant wasn't emitted
/// (e.g. Vulkan SC variants) are skipped because they don't get a
/// `PNextChainable` impl.
fn is_feature_struct(
    def: &StructDefinition,
    known_structure_types: &std::collections::HashSet<String>,
) -> bool {
    if def.name == "VkPhysicalDeviceFeatures" {
        return true;
    }
    if def.deprecated.is_some() || def.is_alias {
        return false;
    }
    let extends_features2 = def
        .structextends
        .as_ref()
        .map(|list| {
            list.split(',')
                .any(|s| s.trim() == "VkPhysicalDeviceFeatures2")
        })
        .unwrap_or(false);
    if !extends_features2 {
        return false;
    }
    // Only include structs whose sType enumerant survived the parser's
    // api-variant filter (main Vulkan, not Vulkan SC).
    let Some(first) = def.members.first() else {
        return false;
    };
    if first.name != "sType" || first.type_name != "VkStructureType" {
        return false;
    }
    let Some(values) = &first.values else {
        return false;
    };
    let stype = values.split(',').next().unwrap_or("").trim();
    known_structure_types.contains(stype)
}

fn load_known_structure_types(
    intermediate_dir: &Path,
) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    let enums_path = intermediate_dir.join("enums.json");
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

/// `true` if a struct member is a `VkBool32` feature bit (not the
/// sType / pNext header).
fn is_bool_feature_bit(m: &StructMember) -> bool {
    if m.name == "sType" || m.name == "pNext" {
        return false;
    }
    m.type_name == "VkBool32"
}

/// A feature bit we will emit a method for.
struct FeatureBit<'a> {
    /// snake_case identifier used as the method name tail (e.g.
    /// `timeline_semaphore`).
    method_name: String,
    /// Original camelCase field name (e.g. `timelineSemaphore`) — used
    /// inside the method body to reach the right struct field.
    field_name: &'a str,
    /// Name of the Vulkan struct the bit lives in. The special value
    /// `"VkPhysicalDeviceFeatures"` is routed through
    /// `features10_mut()`; everything else goes through `ensure_ext::<T>()`.
    struct_name: &'a str,
    /// Doc comment for the generated method (comes from vk.xml if
    /// present, otherwise synthesised).
    doc: String,
}

/// Gather feature bits from all in-scope structs and collapse name
/// collisions by priority.
fn collect_feature_bits<'a>(
    structs: &'a [StructDefinition],
    known_structure_types: &std::collections::HashSet<String>,
) -> Vec<FeatureBit<'a>> {
    use std::collections::HashMap;

    // method_name -> (priority, FeatureBit)
    let mut chosen: HashMap<String, (i32, FeatureBit<'a>)> = HashMap::new();

    for def in structs {
        if !is_feature_struct(def, known_structure_types) {
            continue;
        }
        let prio = struct_priority(&def.name);
        for m in &def.members {
            if !is_bool_feature_bit(m) {
                continue;
            }
            let method_name = camel_to_snake(&m.name);
            let doc = build_doc(def, m);
            let bit = FeatureBit {
                method_name: method_name.clone(),
                field_name: &m.name,
                struct_name: &def.name,
                doc,
            };
            chosen
                .entry(method_name)
                .and_modify(|existing| {
                    if prio > existing.0 {
                        *existing = (
                            prio,
                            FeatureBit {
                                method_name: bit.method_name.clone(),
                                field_name: bit.field_name,
                                struct_name: bit.struct_name,
                                doc: bit.doc.clone(),
                            },
                        );
                    }
                })
                .or_insert((prio, bit));
        }
    }

    let mut bits: Vec<FeatureBit> = chosen.into_iter().map(|(_, (_, v))| v).collect();
    // Stable output order — sort by method name.
    bits.sort_by(|a, b| a.method_name.cmp(&b.method_name));
    bits
}

fn build_doc(def: &StructDefinition, m: &StructMember) -> String {
    // Prefer the vk.xml `comment=` attribute on the member, fall back to
    // a synthesised line pointing at the struct.
    if let Some(c) = &m.comment {
        return crate::codegen::sanitize_doc_line(c);
    }
    format!(
        "Enable `{}` (from [`{struct_name}`](crate::raw::bindings::{struct_name})).",
        m.name,
        struct_name = def.name
    )
}

/// Emit the Rust source of the generated `DeviceFeatures` impl block.
fn emit_impl_block(bits: &[FeatureBit]) -> String {
    let mut out = String::new();
    out.push_str(
        "// Generated by vulkan_gen::device_features_gen — do not edit.\n\
         //\n\
         // Included from `vulkane/src/safe/features.rs`.\n\
         //\n\
         // One `with_<feature>()` method per unique feature bit across every\n\
         // Vulkan struct that extends `VkPhysicalDeviceFeatures2`.\n\n",
    );

    out.push_str("#[allow(non_snake_case)]\n");
    out.push_str("impl DeviceFeatures {\n");
    for bit in bits {
        // Doc comment line
        for line in bit.doc.lines() {
            out.push_str("    /// ");
            out.push_str(line);
            out.push('\n');
        }
        let method = escape_keyword(&format!("with_{}", bit.method_name));
        let field = bit.field_name;
        if bit.struct_name == "VkPhysicalDeviceFeatures" {
            out.push_str(&format!(
                "    pub fn {method}(mut self) -> Self {{\n\
                 \x20       self.features10_mut().{field} = 1;\n\
                 \x20       self\n\
                 \x20   }}\n"
            ));
        } else {
            out.push_str(&format!(
                "    pub fn {method}(mut self) -> Self {{\n\
                 \x20       self.ensure_ext::<crate::raw::bindings::{struct_name}>().{field} = 1;\n\
                 \x20       self\n\
                 \x20   }}\n",
                struct_name = bit.struct_name
            ));
        }
    }
    out.push_str("}\n");
    out
}

/// Generate `device_features_generated.rs` from the parsed structs in
/// `structs.json`, writing it next to the main bindings file.
pub fn generate_device_features(
    intermediate_dir: &Path,
    output_path: &Path,
) -> GeneratorResult<usize> {
    let structs_path = intermediate_dir.join("structs.json");
    let content = fs::read_to_string(&structs_path).map_err(GeneratorError::Io)?;
    let structs: Vec<StructDefinition> = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => {
            // Fallback: object wrapper `{ "structs": [...] }`.
            #[derive(serde::Deserialize)]
            struct Wrapper {
                structs: Vec<StructDefinition>,
            }
            let w: Wrapper = serde_json::from_str(&content).map_err(GeneratorError::Json)?;
            w.structs
        }
    };

    let known = load_known_structure_types(intermediate_dir);
    let bits = collect_feature_bits(&structs, &known);
    let code = emit_impl_block(&bits);

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(GeneratorError::Io)?;
    }
    fs::write(output_path, code).map_err(GeneratorError::Io)?;

    Ok(bits.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_stypes_of(structs: &[StructDefinition]) -> std::collections::HashSet<String> {
        // Accept every sType referenced by the test structs so the
        // sc-filter doesn't silently hide them.
        let mut set = std::collections::HashSet::new();
        for s in structs {
            if let Some(first) = s.members.first()
                && let Some(v) = &first.values
            {
                set.insert(v.split(',').next().unwrap_or("").trim().to_string());
            }
        }
        set
    }

    fn mk_member_with_stype(name: &str, type_name: &str, stype: Option<&str>) -> StructMember {
        let mut m = mk_member(name, type_name);
        m.values = stype.map(|s| s.to_string());
        m
    }

    fn mk_member(name: &str, type_name: &str) -> StructMember {
        StructMember {
            name: name.to_string(),
            type_name: type_name.to_string(),
            optional: None,
            len: None,
            altlen: None,
            noautovalidity: None,
            values: None,
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
        structextends: Option<&str>,
        members: Vec<StructMember>,
    ) -> StructDefinition {
        StructDefinition {
            name: name.to_string(),
            category: "struct".to_string(),
            structextends: structextends.map(|s| s.to_string()),
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

    #[test]
    fn priority_orders_core_above_aggregate_above_extension() {
        assert!(
            struct_priority("VkPhysicalDeviceFeatures")
                > struct_priority("VkPhysicalDeviceVulkan13Features")
        );
        assert!(
            struct_priority("VkPhysicalDeviceVulkan13Features")
                > struct_priority("VkPhysicalDeviceVulkan12Features")
        );
        assert!(
            struct_priority("VkPhysicalDeviceVulkan12Features")
                > struct_priority("VkPhysicalDeviceTimelineSemaphoreFeatures")
        );
        assert!(
            struct_priority("VkPhysicalDeviceTimelineSemaphoreFeatures")
                > struct_priority("VkPhysicalDeviceTimelineSemaphoreFeaturesKHR")
        );
    }

    #[test]
    fn collision_routes_to_highest_priority_struct() {
        let structs = vec![
            mk_struct(
                "VkPhysicalDeviceVulkan12Features",
                Some("VkPhysicalDeviceFeatures2"),
                vec![
                    mk_member_with_stype(
                        "sType",
                        "VkStructureType",
                        Some("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES"),
                    ),
                    mk_member("pNext", "void"),
                    mk_member("timelineSemaphore", "VkBool32"),
                ],
            ),
            mk_struct(
                "VkPhysicalDeviceTimelineSemaphoreFeaturesKHR",
                Some("VkPhysicalDeviceFeatures2"),
                vec![
                    mk_member_with_stype(
                        "sType",
                        "VkStructureType",
                        Some("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES"),
                    ),
                    mk_member("pNext", "void"),
                    mk_member("timelineSemaphore", "VkBool32"),
                ],
            ),
        ];
        let bits = collect_feature_bits(&structs, &known_stypes_of(&structs));
        assert_eq!(bits.len(), 1);
        assert_eq!(bits[0].method_name, "timeline_semaphore");
        assert_eq!(bits[0].struct_name, "VkPhysicalDeviceVulkan12Features");
    }

    #[test]
    fn v10_core_routes_through_features10_mut() {
        let structs = vec![mk_struct(
            "VkPhysicalDeviceFeatures",
            None, // v1.0 core has no structextends
            vec![
                mk_member("robustBufferAccess", "VkBool32"),
                mk_member("samplerAnisotropy", "VkBool32"),
            ],
        )];
        let bits = collect_feature_bits(&structs, &known_stypes_of(&structs));
        assert_eq!(bits.len(), 2);
        let code = emit_impl_block(&bits);
        assert!(code.contains("fn with_robust_buffer_access"));
        assert!(code.contains("self.features10_mut().robustBufferAccess = 1;"));
        assert!(code.contains("fn with_sampler_anisotropy"));
        assert!(code.contains("self.features10_mut().samplerAnisotropy = 1;"));
    }

    #[test]
    fn extension_struct_routes_through_ensure_ext() {
        let structs = vec![mk_struct(
            "VkPhysicalDeviceCooperativeMatrixFeaturesKHR",
            Some("VkPhysicalDeviceFeatures2"),
            vec![
                mk_member_with_stype(
                    "sType",
                    "VkStructureType",
                    Some("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR"),
                ),
                mk_member("pNext", "void"),
                mk_member("cooperativeMatrix", "VkBool32"),
                mk_member("cooperativeMatrixRobustBufferAccess", "VkBool32"),
            ],
        )];
        let bits = collect_feature_bits(&structs, &known_stypes_of(&structs));
        assert_eq!(bits.len(), 2);
        let code = emit_impl_block(&bits);
        assert!(code.contains("pub fn with_cooperative_matrix(mut self) -> Self"));
        assert!(
            code.contains(
                "self.ensure_ext::<crate::raw::bindings::VkPhysicalDeviceCooperativeMatrixFeaturesKHR>().cooperativeMatrix = 1;"
            )
        );
        assert!(code.contains("pub fn with_cooperative_matrix_robust_buffer_access(mut self)"));
    }

    #[test]
    fn non_bool_members_are_skipped() {
        let structs = vec![mk_struct(
            "VkPhysicalDeviceFancyFeaturesEXT",
            Some("VkPhysicalDeviceFeatures2"),
            vec![
                mk_member_with_stype(
                    "sType",
                    "VkStructureType",
                    Some("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FANCY_FEATURES_EXT"),
                ),
                mk_member("pNext", "void"),
                mk_member("fancyEnabled", "VkBool32"),
                mk_member("fancyTunable", "uint32_t"), // should be skipped
            ],
        )];
        let bits = collect_feature_bits(&structs, &known_stypes_of(&structs));
        assert_eq!(bits.len(), 1);
        assert_eq!(bits[0].field_name, "fancyEnabled");
    }

    #[test]
    fn struct_with_missing_stype_is_skipped() {
        // Mirrors the Vulkan SC case: struct is present but its sType
        // enumerant wasn't emitted into the bindings enum. Filter it out
        // so we don't reference a PNextChainable impl that doesn't exist.
        let structs = vec![mk_struct(
            "VkPhysicalDeviceVulkanSC10Features",
            Some("VkPhysicalDeviceFeatures2"),
            vec![
                mk_member_with_stype(
                    "sType",
                    "VkStructureType",
                    Some("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_SC_1_0_FEATURES"),
                ),
                mk_member("pNext", "void"),
                mk_member("shaderAtomicInstructions", "VkBool32"),
            ],
        )];
        // Empty known-set simulates the SC variant being filtered out.
        let known = std::collections::HashSet::new();
        let bits = collect_feature_bits(&structs, &known);
        assert_eq!(bits.len(), 0);
    }

    #[test]
    fn non_feature_structs_are_excluded() {
        // A properties struct that extends VkPhysicalDeviceProperties2,
        // not Features2 — must not contribute any methods.
        let structs = vec![mk_struct(
            "VkPhysicalDeviceSomethingLimits",
            Some("VkPhysicalDeviceProperties2"),
            vec![
                mk_member_with_stype(
                    "sType",
                    "VkStructureType",
                    Some("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SOMETHING_LIMITS"),
                ),
                mk_member("pNext", "void"),
                mk_member("someLimit", "VkBool32"),
            ],
        )];
        let bits = collect_feature_bits(&structs, &known_stypes_of(&structs));
        assert_eq!(bits.len(), 0);
    }
}
