// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vulkan specification data types
//!
//! These types represent parsed Vulkan XML data. They are populated by
//! the tree parser and serialized to intermediate JSON files for the
//! code generator.

use serde::{Deserialize, Serialize};

// ---- Constants ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanConstant {
    pub name: String,
    pub value: Option<String>,
    pub alias: Option<String>,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub constant_type: String,
    pub raw_content: String,
    pub is_alias: bool,
    #[serde(default)]
    pub source_line: Option<usize>,
}

// ---- Enums ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanEnum {
    pub name: String,
    pub enum_type: String,
    pub comment: Option<String>,
    pub bitwidth: Option<String>,
    pub deprecated: Option<String>,
    pub api: Option<String>,
    pub values: Vec<EnumValue>,
    pub raw_content: String,
    pub is_alias: bool,
    #[serde(default)]
    pub source_line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumValue {
    pub name: String,
    pub value: Option<String>,
    pub bitpos: Option<String>,
    pub alias: Option<String>,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub protect: Option<String>,
    pub extnumber: Option<String>,
    pub offset: Option<String>,
    pub dir: Option<String>,
    pub extends: Option<String>,
    pub raw_content: String,
    pub is_alias: bool,
    #[serde(default)]
    pub source_line: Option<usize>,
}

// ---- Structs and Unions ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanStruct {
    pub name: String,
    pub category: String,
    pub comment: Option<String>,
    pub returnedonly: Option<String>,
    pub structextends: Option<String>,
    pub allowduplicate: Option<String>,
    pub deprecated: Option<String>,
    pub alias: Option<String>,
    pub api: Option<String>,
    pub members: Vec<StructMember>,
    pub raw_content: String,
    pub is_alias: bool,
    #[serde(default)]
    pub source_line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructMember {
    pub name: String,
    pub type_name: String,
    pub optional: Option<String>,
    pub len: Option<String>,
    pub altlen: Option<String>,
    pub noautovalidity: Option<String>,
    pub values: Option<String>,
    pub limittype: Option<String>,
    pub selector: Option<String>,
    pub selection: Option<String>,
    pub externsync: Option<String>,
    pub objecttype: Option<String>,
    pub deprecated: Option<String>,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub definition: String,
    pub raw_content: String,
}

// ---- Commands (Functions) ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanCommand {
    pub name: String,
    pub return_type: String,
    pub comment: Option<String>,
    pub successcodes: Option<String>,
    pub errorcodes: Option<String>,
    pub alias: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub cmdbufferlevel: Option<String>,
    pub pipeline: Option<String>,
    pub queues: Option<String>,
    pub renderpass: Option<String>,
    pub videocoding: Option<String>,
    pub parameters: Vec<CommandParam>,
    pub raw_content: String,
    pub is_alias: bool,
    #[serde(default)]
    pub source_line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandParam {
    pub name: String,
    pub type_name: String,
    pub optional: Option<String>,
    pub len: Option<String>,
    pub altlen: Option<String>,
    pub externsync: Option<String>,
    pub noautovalidity: Option<String>,
    pub objecttype: Option<String>,
    pub stride: Option<String>,
    pub validstructs: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub comment: Option<String>,
    pub definition: String,
    pub raw_content: String,
    #[serde(default)]
    pub source_line: Option<usize>,
}

// ---- Types (handles, bitmasks, basetypes, funcpointers) ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanType {
    pub name: String,
    pub category: String,
    pub definition: Option<String>,
    pub api: Option<String>,
    pub requires: Option<String>,
    pub bitvalues: Option<String>,
    pub parent: Option<String>,
    pub objtypeenum: Option<String>,
    pub alias: Option<String>,
    pub deprecated: Option<String>,
    pub comment: Option<String>,
    pub raw_content: String,
    pub type_references: Vec<String>,
    pub is_alias: bool,
}

// ---- Extensions ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanExtension {
    pub name: String,
    pub number: Option<String>,
    pub extension_type: Option<String>,
    pub requires: Option<String>,
    #[serde(rename = "requiresCore")]
    pub requires_core: Option<String>,
    pub author: Option<String>,
    pub contact: Option<String>,
    pub supported: Option<String>,
    pub ratified: Option<String>,
    pub deprecated: Option<String>,
    pub obsoletedby: Option<String>,
    pub promotedto: Option<String>,
    pub provisional: Option<String>,
    pub specialuse: Option<String>,
    pub platform: Option<String>,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub sortorder: Option<String>,
    pub require_blocks: Vec<ExtensionRequire>,
    pub remove_blocks: Vec<ExtensionRemove>,
    pub raw_content: String,
    #[serde(default)]
    pub source_line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtensionRequire {
    pub api: Option<String>,
    pub profile: Option<String>,
    pub extension: Option<String>,
    pub feature: Option<String>,
    pub comment: Option<String>,
    pub depends: Option<String>,
    pub items: Vec<RequireItem>,
    pub raw_content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtensionRemove {
    pub api: Option<String>,
    pub profile: Option<String>,
    pub comment: Option<String>,
    pub items: Vec<RemoveItem>,
    pub raw_content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequireItem {
    pub item_type: String,
    pub name: String,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub value: Option<String>,
    pub bitpos: Option<String>,
    pub offset: Option<String>,
    pub dir: Option<String>,
    pub extends: Option<String>,
    pub extnumber: Option<String>,
    pub alias: Option<String>,
    pub protect: Option<String>,
    pub raw_content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoveItem {
    pub item_type: String,
    pub name: String,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub raw_content: String,
}

// ---- Features ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanFeature {
    pub api: String,
    pub name: String,
    pub number: String,
    pub comment: Option<String>,
    pub deprecated: Option<String>,
    pub require_blocks: Vec<FeatureRequire>,
    pub raw_content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureRequire {
    pub api: Option<String>,
    pub profile: Option<String>,
    pub comment: Option<String>,
    pub items: Vec<FeatureItem>,
    pub raw_content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureItem {
    pub item_type: String,
    pub name: String,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub raw_content: String,
}

// ---- Includes ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanInclude {
    pub filename: String,
    pub category: String,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub raw_content: String,
}

// ---- Macros ----

fn default_macro_type() -> String {
    "object_like".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanMacro {
    pub name: String,
    pub definition: String,
    pub category: String,
    #[serde(default = "default_macro_type")]
    pub macro_type: String,
    pub comment: Option<String>,
    pub deprecated: Option<String>,
    pub requires: Option<String>,
    pub api: Option<String>,
    pub parameters: Vec<String>,
    pub raw_content: String,
    pub parsed_definition: String,
    #[serde(default)]
    pub source_line: Option<usize>,
}

// ---- Platforms ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanPlatform {
    pub name: String,
    pub protect: String,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub raw_content: String,
}

// ---- Tags ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VulkanTag {
    pub name: String,
    pub author: String,
    pub contact: Option<String>,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub deprecated: Option<String>,
    pub raw_content: String,
    pub source_line: Option<usize>,
}

// ---- Compatibility aliases for codegen modules ----
// These map the vulkan-types crate names to our canonical names

pub type ConstantDefinition = VulkanConstant;
pub type EnumDefinition = VulkanEnum;
pub type StructDefinition = VulkanStruct;
pub type TypeDefinition = VulkanType;
pub type FunctionDefinition = VulkanCommand;
pub type FunctionParameter = CommandParam;
pub type ExtensionDefinition = VulkanExtension;
pub type FeatureDefinition = VulkanFeature;
pub type IncludeStatement = VulkanInclude;
pub type MacroDefinition = VulkanMacro;
pub type PlatformDefinition = VulkanPlatform;
pub type TagDefinition = VulkanTag;
pub type ExtensionRequireItem = RequireItem;
pub type ExtensionRemoveItem = RemoveItem;

// Container types (used by type_integration for JSON deserialization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantData {
    pub constants: Vec<VulkanConstant>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumData {
    pub enums: Vec<VulkanEnum>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructData {
    pub structs: Vec<VulkanStruct>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeData {
    pub types: Vec<VulkanType>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionData {
    pub functions: Vec<VulkanCommand>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionData {
    pub extensions: Vec<VulkanExtension>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureData {
    pub features: Vec<VulkanFeature>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncludeData {
    pub includes: Vec<VulkanInclude>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroData {
    pub macros: Vec<VulkanMacro>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformData {
    pub platforms: Vec<VulkanPlatform>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagData {
    pub tags: Vec<VulkanTag>,
}
