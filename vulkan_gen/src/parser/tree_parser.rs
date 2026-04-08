//! Tree-based XML parser for the Vulkan specification
//!
//! Uses `roxmltree` for DOM-based parsing, which correctly handles
//! nested XML elements (unlike the streaming event-bus approach).

use crate::parser::vk_types::*;
use crate::vulkan_spec_parser::VulkanSpecification;

/// Extract the text content of a node, concatenating all descendant text
/// (but not child element text). For mixed-content nodes like:
///   `const <type>VkFoo</type>* <name>pFoo</name>`
/// this returns the full reconstructed text including child element text.
fn full_text(node: roxmltree::Node) -> String {
    let mut result = String::new();
    for child in node.children() {
        if child.is_text() {
            result.push_str(child.text().unwrap_or(""));
        } else if child.is_element() {
            result.push_str(&full_text(child));
        }
    }
    result
}

/// Extract the text content of the first child element with the given tag name
fn child_element_text<'a>(node: roxmltree::Node<'a, 'a>, tag: &str) -> Option<String> {
    node.children()
        .find(|c| c.is_element() && c.tag_name().name() == tag)
        .map(|c| full_text(c))
}

/// Reconstruct the full C declaration from a mixed-content <member> or <param> node.
/// E.g., `const <type>VkFoo</type>* <name>pFoo</name>` → `"const VkFoo* pFoo"`
fn reconstruct_definition(node: roxmltree::Node) -> String {
    let mut parts = Vec::new();
    for child in node.children() {
        if child.is_text() {
            let t = child.text().unwrap_or("");
            if !t.trim().is_empty() {
                parts.push(t.to_string());
            }
        } else if child.is_element() {
            let tag = child.tag_name().name();
            if tag == "type" || tag == "name" || tag == "enum" {
                parts.push(full_text(child));
            }
            // skip <comment> children
        }
    }
    // Join parts and normalize whitespace
    let joined = parts.join(" ");
    // Collapse multiple spaces
    let mut result = String::new();
    let mut prev_space = false;
    for ch in joined.chars() {
        if ch.is_whitespace() {
            if !prev_space && !result.is_empty() {
                result.push(' ');
                prev_space = true;
            }
        } else {
            // Fix "* " → "*" adjacency: remove space before/after * for pointer declarations
            result.push(ch);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

/// Get the raw XML content of a node as a string (for raw_content fields)
fn raw_xml_content(node: roxmltree::Node) -> String {
    full_text(node)
}

fn attr(node: roxmltree::Node, name: &str) -> Option<String> {
    node.attribute(name).map(|s| s.to_string())
}

/// Get a documentation comment from a node, checking both the `comment` attribute
/// and a `<comment>` child element. Returns None if neither is present.
fn comment_or_child(node: roxmltree::Node) -> Option<String> {
    if let Some(c) = node.attribute("comment") {
        return Some(c.to_string());
    }
    node.children()
        .find(|c| c.is_element() && c.tag_name().name() == "comment")
        .and_then(|c| c.text())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Parse the complete Vulkan specification from an XML string using DOM parsing
pub fn parse_vk_xml(xml_content: &str) -> Result<VulkanSpecification, String> {
    let doc =
        roxmltree::Document::parse(xml_content).map_err(|e| format!("XML parse error: {}", e))?;
    let root = doc.root_element(); // <registry>

    let mut spec = VulkanSpecification::default();

    for child in root.children().filter(|n| n.is_element()) {
        // Top-level api-profile filter: any element tagged for a non-
        // desktop Vulkan profile is skipped entirely. This catches
        // <feature api="vulkansc"> blocks (which would otherwise
        // contribute Vulkan SC core enum values to spock's enums) and
        // any other top-level element that gets tagged in future
        // vk.xml releases.
        if let Some(api) = attr(child, "api") {
            let included = api
                .split(',')
                .any(|s| matches!(s.trim(), "vulkan" | "vulkanbase"));
            if !included {
                continue;
            }
        }

        match child.tag_name().name() {
            "types" => parse_types_section(child, &mut spec),
            "enums" => parse_enums_block(child, &mut spec),
            "commands" => parse_commands_section(child, &mut spec),
            "extensions" => parse_extensions_section(child, &mut spec),
            "feature" => parse_feature(child, &mut spec),
            "platforms" => parse_platforms_section(child, &mut spec),
            "tags" => parse_tags_section(child, &mut spec),
            _ => {} // comment, spirvextensions, spirvcapabilities, sync, formats, etc.
        }
    }

    // Post-processing: merge extension enum values into base enums
    // (Feature enum values are merged inline during parse_feature)
    merge_extension_enum_values(&mut spec);

    Ok(spec)
}

/// Compute the numeric value for an extension enum entry using the Vulkan formula:
/// base_value = 1000000000 + (extnumber - 1) * 1000 + offset
/// If dir == "-", negate the result.
fn compute_extension_enum_value(extnumber: &str, offset: &str, dir: Option<&str>) -> String {
    let ext_num: i64 = extnumber.parse().unwrap_or(0);
    let off: i64 = offset.parse().unwrap_or(0);
    let mut value = 1000000000 + (ext_num - 1) * 1000 + off;
    if dir == Some("-") {
        value = -value;
    }
    value.to_string()
}

/// Merge extension enum values (from extensions' require blocks) into base enum definitions
fn merge_extension_enum_values(spec: &mut VulkanSpecification) {
    // Collect all extension enum items that extend a base enum
    let mut additions: Vec<(String, EnumValue)> = Vec::new();

    for ext in &spec.extensions {
        let ext_number = ext.number.as_deref().unwrap_or("0");

        for req_block in &ext.require_blocks {
            for item in &req_block.items {
                if item.item_type != "enum" {
                    continue;
                }
                let extends = match &item.extends {
                    Some(e) => e.clone(),
                    None => continue, // Not extending any enum
                };

                // Compute the value. Note: bitpos values and alias references
                // are stored in the EnumValue struct's separate fields, not in
                // `value`, so we leave `value` as None for those cases.
                let value = if let Some(v) = &item.value {
                    Some(v.clone())
                } else if let Some(offset) = &item.offset {
                    let ext_num = item.extnumber.as_deref().unwrap_or(ext_number);
                    Some(compute_extension_enum_value(
                        ext_num,
                        offset,
                        item.dir.as_deref(),
                    ))
                } else if item.bitpos.is_some() || item.alias.is_some() {
                    None
                } else {
                    continue;
                };

                additions.push((
                    extends,
                    EnumValue {
                        name: item.name.clone(),
                        value,
                        bitpos: item.bitpos.clone(),
                        alias: item.alias.clone(),
                        comment: item.comment.clone(),
                        api: item.api.clone(),
                        deprecated: item.deprecated.clone(),
                        protect: item.protect.clone(),
                        extnumber: item.extnumber.clone(),
                        offset: item.offset.clone(),
                        dir: item.dir.clone(),
                        extends: item.extends.clone(),
                        raw_content: item.raw_content.clone(),
                        is_alias: item.alias.is_some(),
                        source_line: None,
                    },
                ));
            }
        }
    }

    // Apply additions to base enums
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    // First collect existing enum value names to avoid duplicates
    for e in &spec.enums {
        for v in &e.values {
            seen_names.insert(v.name.clone());
        }
    }

    for (target_enum, value) in additions {
        if seen_names.contains(&value.name) {
            continue;
        }
        seen_names.insert(value.name.clone());

        if let Some(base_enum) = spec.enums.iter_mut().find(|e| e.name == target_enum) {
            base_enum.values.push(value);
        }
    }
}

// Feature enum extensions are handled inline during parse_feature()

// ---------------------------------------------------------------------------
// Types section: <types> contains <type category="..."> elements
// ---------------------------------------------------------------------------

fn parse_types_section(types_node: roxmltree::Node, spec: &mut VulkanSpecification) {
    for type_node in types_node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "type")
    {
        // Filter on the `api` attribute the same way commands and
        // struct members do — vk.xml ships some `<type>` definitions
        // for both desktop Vulkan and Vulkan SC (Safety Critical),
        // and the SC variants would otherwise leak into the generated
        // bindings as duplicate or wrong definitions. Skip any type
        // explicitly tagged with a non-desktop profile. Types with no
        // `api` attribute apply universally and are kept.
        if let Some(api) = attr(type_node, "api") {
            let included = api
                .split(',')
                .any(|s| matches!(s.trim(), "vulkan" | "vulkanbase"));
            if !included {
                continue;
            }
        }

        let category = attr(type_node, "category").unwrap_or_default();

        match category.as_str() {
            "struct" | "union" => parse_struct_or_union(type_node, &category, spec),
            "include" => parse_include(type_node, spec),
            "define" => parse_define(type_node, spec),
            _ => parse_general_type(type_node, &category, spec),
        }
    }
}

fn parse_struct_or_union(node: roxmltree::Node, category: &str, spec: &mut VulkanSpecification) {
    let name = attr(node, "name").unwrap_or_default();

    // Check for alias
    if let Some(alias) = attr(node, "alias") {
        spec.structs.push(VulkanStruct {
            name,
            category: category.to_string(),
            comment: comment_or_child(node),
            returnedonly: attr(node, "returnedonly"),
            structextends: attr(node, "structextends"),
            allowduplicate: attr(node, "allowduplicate"),
            deprecated: attr(node, "deprecated"),
            alias: Some(alias),
            api: attr(node, "api"),
            members: Vec::new(),
            raw_content: raw_xml_content(node),
            is_alias: true,
            source_line: None,
        });
        return;
    }

    let mut members = Vec::new();
    for member_node in node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "member")
    {
        // Filter on `api` attribute the same way commands do — vk.xml
        // sometimes lists the same struct member twice for the desktop
        // Vulkan profile vs Vulkan SC. Skip non-desktop entries.
        if let Some(api) = attr(member_node, "api") {
            let included = api
                .split(',')
                .any(|s| matches!(s.trim(), "vulkan" | "vulkanbase"));
            if !included {
                continue;
            }
        }
        let member_name = child_element_text(member_node, "name").unwrap_or_default();
        let type_name = child_element_text(member_node, "type").unwrap_or_default();
        let definition = reconstruct_definition(member_node);

        members.push(StructMember {
            name: member_name,
            type_name,
            optional: attr(member_node, "optional"),
            len: attr(member_node, "len"),
            altlen: attr(member_node, "altlen"),
            noautovalidity: attr(member_node, "noautovalidity"),
            values: attr(member_node, "values"),
            limittype: attr(member_node, "limittype"),
            selector: attr(member_node, "selector"),
            selection: attr(member_node, "selection"),
            externsync: attr(member_node, "externsync"),
            objecttype: attr(member_node, "objecttype"),
            deprecated: attr(member_node, "deprecated"),
            comment: comment_or_child(member_node),
            api: attr(member_node, "api"),
            definition,
            raw_content: raw_xml_content(member_node),
        });
    }

    spec.structs.push(VulkanStruct {
        name,
        category: category.to_string(),
        comment: comment_or_child(node),
        returnedonly: attr(node, "returnedonly"),
        structextends: attr(node, "structextends"),
        allowduplicate: attr(node, "allowduplicate"),
        deprecated: attr(node, "deprecated"),
        alias: None,
        api: attr(node, "api"),
        members,
        raw_content: raw_xml_content(node),
        is_alias: false,
        source_line: None,
    });
}

fn parse_include(node: roxmltree::Node, spec: &mut VulkanSpecification) {
    let name = attr(node, "name").unwrap_or_default();
    if name.is_empty() {
        return;
    }
    spec.includes.push(VulkanInclude {
        filename: name,
        category: "include".to_string(),
        comment: attr(node, "comment"),
        api: attr(node, "api"),
        deprecated: attr(node, "deprecated"),
        raw_content: raw_xml_content(node),
    });
}

fn parse_define(node: roxmltree::Node, spec: &mut VulkanSpecification) {
    let name = attr(node, "name")
        .or_else(|| child_element_text(node, "name"))
        .unwrap_or_default();
    if name.is_empty() {
        return;
    }

    let raw = raw_xml_content(node);

    // Determine macro type and extract parameters
    let (macro_type, parameters) = if raw.contains('(') && raw.contains(')') {
        // Function-like macro: extract parameter names
        let params = extract_macro_params(&raw);
        ("function_like".to_string(), params)
    } else {
        ("object_like".to_string(), Vec::new())
    };

    spec.macros.push(VulkanMacro {
        name: name.clone(),
        definition: raw.clone(),
        category: "define".to_string(),
        macro_type,
        comment: attr(node, "comment"),
        deprecated: attr(node, "deprecated"),
        requires: attr(node, "requires"),
        api: attr(node, "api"),
        parameters,
        raw_content: raw.clone(),
        parsed_definition: raw,
        source_line: None,
    });
}

fn extract_macro_params(definition: &str) -> Vec<String> {
    // Find content between first ( and matching )
    if let Some(start) = definition.find('(') {
        if let Some(end) = definition[start..].find(')') {
            let params_str = &definition[start + 1..start + end];
            return params_str
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect();
        }
    }
    Vec::new()
}

fn type_already_exists(spec: &VulkanSpecification, name: &str) -> bool {
    spec.types.iter().any(|t| t.name == name)
        || spec.structs.iter().any(|s| s.name == name)
        || spec.enums.iter().any(|e| e.name == name)
}

fn parse_general_type(node: roxmltree::Node, category: &str, spec: &mut VulkanSpecification) {
    // Try multiple locations for the type name:
    // 1. name= attribute
    // 2. Direct <name> child element
    // 3. <proto>/<name> (newer funcpointer format)
    let name = attr(node, "name")
        .or_else(|| child_element_text(node, "name"))
        .or_else(|| {
            // Check <proto>/<name> for newer funcpointer format
            node.children()
                .find(|c| c.is_element() && c.tag_name().name() == "proto")
                .and_then(|proto| child_element_text(proto, "name"))
        })
        .unwrap_or_default();
    if name.is_empty() {
        return;
    }

    // Skip duplicates
    if type_already_exists(spec, &name) {
        return;
    }

    // Check for alias
    let is_alias = attr(node, "alias").is_some();

    // Extract type references from child <type> elements
    let type_refs: Vec<String> = node
        .children()
        .filter(|c| c.is_element() && c.tag_name().name() == "type")
        .map(|c| full_text(c))
        .collect();

    // Build definition from full text content
    let definition = {
        let text = raw_xml_content(node);
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    };

    spec.types.push(VulkanType {
        name,
        category: category.to_string(),
        definition,
        api: attr(node, "api"),
        requires: attr(node, "requires"),
        bitvalues: attr(node, "bitvalues"),
        parent: attr(node, "parent"),
        objtypeenum: attr(node, "objtypeenum"),
        alias: attr(node, "alias"),
        deprecated: attr(node, "deprecated"),
        comment: comment_or_child(node),
        raw_content: raw_xml_content(node),
        type_references: type_refs,
        is_alias,
    });
}

// ---------------------------------------------------------------------------
// Enums: <enums name="..." type="..."> blocks
// ---------------------------------------------------------------------------

fn parse_enums_block(node: roxmltree::Node, spec: &mut VulkanSpecification) {
    let name = attr(node, "name").unwrap_or_default();
    let enum_type = attr(node, "type").unwrap_or_default();

    // "API Constants" block contains individual constants, not an enum
    if name == "API Constants" || enum_type.is_empty() {
        for enum_child in node
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "enum")
        {
            let const_name = attr(enum_child, "name").unwrap_or_default();
            let is_alias = attr(enum_child, "alias").is_some();
            spec.constants.push(VulkanConstant {
                name: const_name,
                value: attr(enum_child, "value"),
                alias: attr(enum_child, "alias"),
                comment: attr(enum_child, "comment"),
                api: attr(enum_child, "api"),
                deprecated: attr(enum_child, "deprecated"),
                constant_type: attr(enum_child, "type").unwrap_or_else(|| "enum".to_string()),
                raw_content: raw_xml_content(enum_child),
                is_alias,
                source_line: None,
            });
        }
        return;
    }

    // Regular enum or bitmask
    let mut values = Vec::new();
    for enum_child in node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "enum")
    {
        let is_alias = attr(enum_child, "alias").is_some();
        values.push(EnumValue {
            name: attr(enum_child, "name").unwrap_or_default(),
            value: attr(enum_child, "value"),
            bitpos: attr(enum_child, "bitpos"),
            alias: attr(enum_child, "alias"),
            comment: attr(enum_child, "comment"),
            api: attr(enum_child, "api"),
            deprecated: attr(enum_child, "deprecated"),
            protect: attr(enum_child, "protect"),
            extnumber: None,
            offset: None,
            dir: None,
            extends: None,
            raw_content: raw_xml_content(enum_child),
            is_alias,
            source_line: None,
        });
    }

    spec.enums.push(VulkanEnum {
        name,
        enum_type,
        comment: attr(node, "comment"),
        bitwidth: attr(node, "bitwidth"),
        deprecated: attr(node, "deprecated"),
        api: attr(node, "api"),
        values,
        raw_content: raw_xml_content(node),
        is_alias: false,
        source_line: None,
    });
}

// ---------------------------------------------------------------------------
// Commands: <commands> → <command>
// ---------------------------------------------------------------------------

fn parse_commands_section(commands_node: roxmltree::Node, spec: &mut VulkanSpecification) {
    for cmd_node in commands_node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "command")
    {
        // Filter out commands explicitly tagged for a non-desktop API
        // profile (Vulkan SC). The same filter applies symmetrically to
        // <param> children inside parse_command.
        if let Some(api) = attr(cmd_node, "api") {
            let included = api
                .split(',')
                .any(|s| matches!(s.trim(), "vulkan" | "vulkanbase"));
            if !included {
                continue;
            }
        }

        // Skip duplicate commands (some appear twice for vulkan/vulkansc APIs)
        let name = cmd_node
            .children()
            .find(|n| n.is_element() && n.tag_name().name() == "proto")
            .and_then(|proto| child_element_text(proto, "name"))
            .or_else(|| attr(cmd_node, "name"));
        if let Some(ref n) = name {
            if spec.functions.iter().any(|f| f.name == *n) {
                continue;
            }
        }
        parse_command(cmd_node, spec);
    }
}

fn parse_command(node: roxmltree::Node, spec: &mut VulkanSpecification) {
    // Alias commands: <command name="vkFoo" alias="vkBar"/>
    if let Some(alias) = attr(node, "alias") {
        let name = attr(node, "name").unwrap_or_default();
        spec.functions.push(VulkanCommand {
            name,
            return_type: String::new(),
            comment: attr(node, "comment"),
            successcodes: attr(node, "successcodes"),
            errorcodes: attr(node, "errorcodes"),
            alias: Some(alias),
            api: attr(node, "api"),
            deprecated: attr(node, "deprecated"),
            cmdbufferlevel: attr(node, "cmdbufferlevel"),
            pipeline: attr(node, "pipeline"),
            queues: attr(node, "queues"),
            renderpass: attr(node, "renderpass"),
            videocoding: attr(node, "videocoding"),
            parameters: Vec::new(),
            raw_content: raw_xml_content(node),
            is_alias: true,
            source_line: None,
        });
        return;
    }

    // Full command definition: <command><proto>...</proto><param>...</param>...</command>
    let proto = node
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == "proto");

    let (name, return_type) = if let Some(proto_node) = proto {
        let cmd_name = child_element_text(proto_node, "name").unwrap_or_default();
        let ret_type = child_element_text(proto_node, "type").unwrap_or_default();
        (cmd_name, ret_type)
    } else {
        (String::new(), String::new())
    };

    let mut parameters = Vec::new();
    for param_node in node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "param")
    {
        // Filter on `api` attribute: vk.xml sometimes lists the same
        // parameter twice for the desktop Vulkan profile vs Vulkan SC
        // (Safety Critical). spock targets desktop Vulkan, so skip
        // entries explicitly tagged for a non-desktop profile. Entries
        // with no `api` attribute apply to all profiles and are kept.
        if let Some(api) = attr(param_node, "api") {
            // The attribute is a comma-separated list. Keep the param
            // only if it lists "vulkan" or "vulkanbase".
            let included = api
                .split(',')
                .any(|s| matches!(s.trim(), "vulkan" | "vulkanbase"));
            if !included {
                continue;
            }
        }
        let param_name = child_element_text(param_node, "name").unwrap_or_default();
        let type_name = child_element_text(param_node, "type").unwrap_or_default();
        let definition = reconstruct_definition(param_node);

        parameters.push(CommandParam {
            name: param_name,
            type_name,
            optional: attr(param_node, "optional"),
            len: attr(param_node, "len"),
            altlen: attr(param_node, "altlen"),
            externsync: attr(param_node, "externsync"),
            noautovalidity: attr(param_node, "noautovalidity"),
            objecttype: attr(param_node, "objecttype"),
            stride: attr(param_node, "stride"),
            validstructs: attr(param_node, "validstructs"),
            api: attr(param_node, "api"),
            deprecated: attr(param_node, "deprecated"),
            comment: attr(param_node, "comment"),
            definition,
            raw_content: raw_xml_content(param_node),
            source_line: None,
        });
    }

    spec.functions.push(VulkanCommand {
        name,
        return_type,
        comment: attr(node, "comment"),
        successcodes: attr(node, "successcodes"),
        errorcodes: attr(node, "errorcodes"),
        alias: None,
        api: attr(node, "api"),
        deprecated: attr(node, "deprecated"),
        cmdbufferlevel: attr(node, "cmdbufferlevel"),
        pipeline: attr(node, "pipeline"),
        queues: attr(node, "queues"),
        renderpass: attr(node, "renderpass"),
        videocoding: attr(node, "videocoding"),
        parameters,
        raw_content: raw_xml_content(node),
        is_alias: false,
        source_line: None,
    });
}

// ---------------------------------------------------------------------------
// Extensions: <extensions> → <extension>
// ---------------------------------------------------------------------------

fn parse_extensions_section(extensions_node: roxmltree::Node, spec: &mut VulkanSpecification) {
    for ext_node in extensions_node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "extension")
    {
        parse_extension(ext_node, spec);
    }
}

fn parse_extension(node: roxmltree::Node, spec: &mut VulkanSpecification) {
    let name = attr(node, "name").unwrap_or_default();
    let number = attr(node, "number");

    let mut require_blocks = Vec::new();
    let mut remove_blocks = Vec::new();

    for child in node.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "require" => {
                require_blocks.push(parse_require_block(child, &number));
            }
            "remove" => {
                remove_blocks.push(parse_remove_block(child));
            }
            _ => {}
        }
    }

    spec.extensions.push(VulkanExtension {
        name,
        number,
        extension_type: attr(node, "type"),
        requires: attr(node, "requires"),
        requires_core: attr(node, "requiresCore"),
        author: attr(node, "author"),
        contact: attr(node, "contact"),
        supported: attr(node, "supported"),
        ratified: attr(node, "ratified"),
        deprecated: attr(node, "deprecated"),
        obsoletedby: attr(node, "obsoletedby"),
        promotedto: attr(node, "promotedto"),
        provisional: attr(node, "provisional"),
        specialuse: attr(node, "specialuse"),
        platform: attr(node, "platform"),
        comment: attr(node, "comment"),
        api: attr(node, "api"),
        sortorder: attr(node, "sortorder"),
        require_blocks,
        remove_blocks,
        raw_content: raw_xml_content(node),
        source_line: None,
    });
}

fn parse_require_block(node: roxmltree::Node, ext_number: &Option<String>) -> ExtensionRequire {
    let mut items = Vec::new();

    for child in node.children().filter(|n| n.is_element()) {
        // Per-item api filter inside <require> blocks. Skip items
        // explicitly tagged for non-desktop Vulkan profiles.
        if let Some(api) = attr(child, "api") {
            let included = api
                .split(',')
                .any(|s| matches!(s.trim(), "vulkan" | "vulkanbase"));
            if !included {
                continue;
            }
        }

        let tag = child.tag_name().name();
        let item_type = match tag {
            "command" => "command",
            "type" => "type",
            "enum" => "enum",
            "comment" => continue,
            _ => continue,
        };

        let name = attr(child, "name").unwrap_or_default();
        if name.is_empty() {
            continue;
        }

        // For enum extensions, use the extension's number if not specified on the item
        let extnumber = attr(child, "extnumber").or_else(|| ext_number.clone());

        items.push(RequireItem {
            item_type: item_type.to_string(),
            name,
            comment: attr(child, "comment"),
            api: attr(child, "api"),
            deprecated: attr(child, "deprecated"),
            value: attr(child, "value"),
            bitpos: attr(child, "bitpos"),
            offset: attr(child, "offset"),
            dir: attr(child, "dir"),
            extends: attr(child, "extends"),
            extnumber,
            alias: attr(child, "alias"),
            protect: attr(child, "protect"),
            raw_content: raw_xml_content(child),
        });
    }

    ExtensionRequire {
        api: attr(node, "api"),
        profile: attr(node, "profile"),
        extension: attr(node, "extension"),
        feature: attr(node, "feature"),
        comment: attr(node, "comment"),
        depends: attr(node, "depends"),
        items,
        raw_content: raw_xml_content(node),
    }
}

fn parse_remove_block(node: roxmltree::Node) -> ExtensionRemove {
    let mut items = Vec::new();

    for child in node.children().filter(|n| n.is_element()) {
        let tag = child.tag_name().name();
        let item_type = match tag {
            "command" => "command",
            "type" => "type",
            "enum" => "enum",
            _ => continue,
        };

        items.push(RemoveItem {
            item_type: item_type.to_string(),
            name: attr(child, "name").unwrap_or_default(),
            comment: attr(child, "comment"),
            api: attr(child, "api"),
            raw_content: raw_xml_content(child),
        });
    }

    ExtensionRemove {
        api: attr(node, "api"),
        profile: attr(node, "profile"),
        comment: attr(node, "comment"),
        items,
        raw_content: raw_xml_content(node),
    }
}

// ---------------------------------------------------------------------------
// Features: <feature api="..." name="..." number="...">
// ---------------------------------------------------------------------------

fn parse_feature(node: roxmltree::Node, spec: &mut VulkanSpecification) {
    let mut require_blocks = Vec::new();

    // Collect enum extension items from feature require blocks.
    // These extend base enums (e.g., VkStructureType) with values
    // for core API versions (Vulkan 1.1, 1.2, 1.3, 1.4).
    let mut enum_extensions: Vec<(String, EnumValue)> = Vec::new();

    for child in node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "require")
    {
        let mut items = Vec::new();
        for item_node in child.children().filter(|n| n.is_element()) {
            let tag = item_node.tag_name().name();
            let item_type = match tag {
                "command" => "command",
                "type" => "type",
                "enum" => {
                    // Check if this enum item extends a base enum
                    if let Some(extends) = attr(item_node, "extends") {
                        let value = if let Some(v) = attr(item_node, "value") {
                            Some(v)
                        } else if let Some(offset) = attr(item_node, "offset") {
                            let ext_num =
                                attr(item_node, "extnumber").unwrap_or_else(|| "0".to_string());
                            Some(compute_extension_enum_value(
                                &ext_num,
                                &offset,
                                attr(item_node, "dir").as_deref(),
                            ))
                        } else {
                            None
                        };

                        enum_extensions.push((
                            extends,
                            EnumValue {
                                name: attr(item_node, "name").unwrap_or_default(),
                                value,
                                bitpos: attr(item_node, "bitpos"),
                                alias: attr(item_node, "alias"),
                                comment: attr(item_node, "comment"),
                                api: attr(item_node, "api"),
                                deprecated: attr(item_node, "deprecated"),
                                protect: attr(item_node, "protect"),
                                extnumber: attr(item_node, "extnumber"),
                                offset: attr(item_node, "offset"),
                                dir: attr(item_node, "dir"),
                                extends: attr(item_node, "extends"),
                                raw_content: raw_xml_content(item_node),
                                is_alias: attr(item_node, "alias").is_some(),
                                source_line: None,
                            },
                        ));
                        continue; // Don't add to feature items
                    }
                    "enum"
                }
                "comment" => continue,
                _ => continue,
            };

            items.push(FeatureItem {
                item_type: item_type.to_string(),
                name: attr(item_node, "name").unwrap_or_default(),
                comment: attr(item_node, "comment"),
                api: attr(item_node, "api"),
                deprecated: attr(item_node, "deprecated"),
                raw_content: raw_xml_content(item_node),
            });
        }

        require_blocks.push(FeatureRequire {
            api: attr(child, "api"),
            profile: attr(child, "profile"),
            comment: attr(child, "comment"),
            items,
            raw_content: raw_xml_content(child),
        });
    }

    spec.features.push(VulkanFeature {
        api: attr(node, "api").unwrap_or_default(),
        name: attr(node, "name").unwrap_or_default(),
        number: attr(node, "number").unwrap_or_default(),
        comment: attr(node, "comment"),
        deprecated: attr(node, "deprecated"),
        require_blocks,
        raw_content: raw_xml_content(node),
    });

    // Merge feature enum extensions into base enums
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for e in &spec.enums {
        for v in &e.values {
            seen.insert(v.name.clone());
        }
    }
    for (target, value) in enum_extensions {
        if seen.contains(&value.name) {
            continue;
        }
        seen.insert(value.name.clone());
        if let Some(base_enum) = spec.enums.iter_mut().find(|e| e.name == target) {
            base_enum.values.push(value);
        }
    }
}

// ---------------------------------------------------------------------------
// Platforms: <platforms> → <platform>
// ---------------------------------------------------------------------------

fn parse_platforms_section(node: roxmltree::Node, spec: &mut VulkanSpecification) {
    for platform in node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "platform")
    {
        spec.platforms.push(VulkanPlatform {
            name: attr(platform, "name").unwrap_or_default(),
            protect: attr(platform, "protect").unwrap_or_default(),
            comment: attr(platform, "comment"),
            api: attr(platform, "api"),
            deprecated: attr(platform, "deprecated"),
            raw_content: raw_xml_content(platform),
        });
    }
}

// ---------------------------------------------------------------------------
// Tags: <tags> → <tag>
// ---------------------------------------------------------------------------

fn parse_tags_section(node: roxmltree::Node, spec: &mut VulkanSpecification) {
    for tag in node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "tag")
    {
        spec.tags.push(VulkanTag {
            name: attr(tag, "name").unwrap_or_default(),
            author: attr(tag, "author").unwrap_or_default(),
            contact: attr(tag, "contact"),
            comment: attr(tag, "comment"),
            api: attr(tag, "api"),
            deprecated: attr(tag, "deprecated"),
            raw_content: raw_xml_content(tag),
            source_line: None,
        });
    }
}
