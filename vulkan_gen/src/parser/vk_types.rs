// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vulkan specification data types
//!
//! These types represent parsed Vulkan XML data. They are populated by
//! the tree parser and serialized to intermediate JSON files for the
//! code generator.

// The fields of these types mirror vk.xml attributes one to one and the parser carries no per-field
// registry text to document them from, so they are deliberately left undocumented (ROADMAP row 9a).
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};

// ---- Constants ----

/// One API constant: an `<enum>` entry of the `API Constants` block of vk.xml.
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

/// One `<enums>` block of vk.xml: an enum or a bitmask, with its values.
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

/// One `<enum>` value inside an `<enums>` block, or one enum value an extension or feature adds.
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

/// One struct or union `<type>` of vk.xml, with its members.
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

/// One `<member>` of a struct or union.
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

/// One `<command>` of vk.xml: its return type, parameters and the error and success codes it declares.
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

/// One `<param>` of a command.
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

/// One `<type>` of vk.xml that is not a struct or union (handle, bitmask, base type, function pointer, ...), distinguished by `category`.
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

/// One `<extension>` of vk.xml, with its require and remove blocks.
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

/// One `<require>` block of an extension: the items it adds, and the extension, feature or `depends` expression it is conditional on.
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

/// One `<remove>` block of an extension: the items it removes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtensionRemove {
    pub api: Option<String>,
    pub profile: Option<String>,
    pub comment: Option<String>,
    pub items: Vec<RemoveItem>,
    pub raw_content: String,
}

/// One item (a type, command, enum or constant) named by a `<require>` block; `item_type` says which.
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

/// One item named by a `<remove>` block; `item_type` says which.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoveItem {
    pub item_type: String,
    pub name: String,
    pub comment: Option<String>,
    pub api: Option<String>,
    pub raw_content: String,
}

// ---- Features ----

/// One core-version `<feature>` of vk.xml (for example Vulkan 1.1), with its require blocks.
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

/// One `<require>` block of a core-version feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureRequire {
    pub api: Option<String>,
    pub profile: Option<String>,
    pub comment: Option<String>,
    pub items: Vec<FeatureItem>,
    pub raw_content: String,
}

/// One item named by a core-version feature's `<require>` block; `item_type` says which.
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

/// One `#include` that vk.xml records for the generated header.
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

/// One C macro that vk.xml records, with its definition and parameters.
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

/// One `<platform>` of vk.xml: a platform name and the preprocessor guard (`protect`) for it.
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

/// One vendor `<tag>` of vk.xml: an author tag such as `KHR` or `EXT`, and its contact.
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

/// Compatibility name for [`VulkanConstant`], used by the code-generation modules.
pub type ConstantDefinition = VulkanConstant;
/// Compatibility name for [`VulkanEnum`], used by the code-generation modules.
pub type EnumDefinition = VulkanEnum;
/// Compatibility name for [`VulkanStruct`], used by the code-generation modules.
pub type StructDefinition = VulkanStruct;
/// Compatibility name for [`VulkanType`], used by the code-generation modules.
pub type TypeDefinition = VulkanType;
/// Compatibility name for [`VulkanCommand`], used by the code-generation modules.
pub type FunctionDefinition = VulkanCommand;
/// Compatibility name for [`CommandParam`], used by the code-generation modules.
pub type FunctionParameter = CommandParam;
/// Compatibility name for [`VulkanExtension`], used by the code-generation modules.
pub type ExtensionDefinition = VulkanExtension;
/// Compatibility name for [`VulkanFeature`], used by the code-generation modules.
pub type FeatureDefinition = VulkanFeature;
/// Compatibility name for [`VulkanInclude`], used by the code-generation modules.
pub type IncludeStatement = VulkanInclude;
/// Compatibility name for [`VulkanMacro`], used by the code-generation modules.
pub type MacroDefinition = VulkanMacro;
/// Compatibility name for [`VulkanPlatform`], used by the code-generation modules.
pub type PlatformDefinition = VulkanPlatform;
/// Compatibility name for [`VulkanTag`], used by the code-generation modules.
pub type TagDefinition = VulkanTag;
/// Compatibility name for [`RequireItem`], used by the code-generation modules.
pub type ExtensionRequireItem = RequireItem;
/// Compatibility name for [`RemoveItem`], used by the code-generation modules.
pub type ExtensionRemoveItem = RemoveItem;

// Container types (used by type_integration for JSON deserialization)
/// The on-disk shape of `constants.json`: the list of constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantData {
    pub constants: Vec<VulkanConstant>,
}
/// The on-disk shape of `enums.json`: the list of enums.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumData {
    pub enums: Vec<VulkanEnum>,
}
/// The on-disk shape of `structs.json`: the list of structs and unions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructData {
    pub structs: Vec<VulkanStruct>,
}
/// The on-disk shape of `types.json`: the list of non-struct types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeData {
    pub types: Vec<VulkanType>,
}
/// The on-disk shape of `functions.json`: the list of commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionData {
    pub functions: Vec<VulkanCommand>,
}
/// The on-disk shape of `extensions.json`: the list of extensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionData {
    pub extensions: Vec<VulkanExtension>,
}
/// The on-disk shape of `features.json`: the list of core-version features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureData {
    pub features: Vec<VulkanFeature>,
}
/// The on-disk shape of `includes.json`: the list of includes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncludeData {
    pub includes: Vec<VulkanInclude>,
}
/// The on-disk shape of `macros.json`: the list of macros.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroData {
    pub macros: Vec<VulkanMacro>,
}
/// The on-disk shape of `platforms.json`: the list of platforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformData {
    pub platforms: Vec<VulkanPlatform>,
}
/// The on-disk shape of `tags.json`: the list of vendor tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagData {
    pub tags: Vec<VulkanTag>,
}

/// Complete Vulkan specification data
#[derive(Debug, Default)]
pub struct VulkanSpecification {
    pub constants: Vec<VulkanConstant>,
    pub enums: Vec<VulkanEnum>,
    pub structs: Vec<VulkanStruct>,
    pub types: Vec<VulkanType>,
    pub extensions: Vec<VulkanExtension>,
    pub functions: Vec<VulkanCommand>,
    pub features: Vec<VulkanFeature>,
    pub includes: Vec<VulkanInclude>,
    pub macros: Vec<VulkanMacro>,
    pub platforms: Vec<VulkanPlatform>,
    pub tags: Vec<VulkanTag>,
}
