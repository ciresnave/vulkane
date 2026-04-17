//! Generator for the safe auto-RAII layer.
//!
//! Emits one typed handle wrapper per Vulkan handle type that is
//! **not** already covered by a hand-written `safe` module, with a
//! `create(…)` constructor derived from the paired `vkCreate<H>` /
//! `vkAllocate<H>` command and a `Drop` impl that calls the paired
//! `vkDestroy<H>` / `vkFree<H>` command. The output is a single
//! `auto_handles_generated.rs` file that's `include!`'d from
//! `vulkane/src/safe/auto.rs`.
//!
//! # Scope
//!
//! Phase-1 of the auto-safe-layer rollout: only handles that fit the
//! **simple Create/Destroy** pattern are generated —
//!
//! ```text
//! VkResult vkCreate<H>(VkDevice, *const VkCreateInfoH, *const VkAllocationCallbacks, *mut VkH);
//! void     vkDestroy<H>(VkDevice, VkH, *const VkAllocationCallbacks);
//! ```
//!
//! Handles that don't fit (pool-allocated, multi-create, instance-
//! parented with non-Device destroy) are skipped for now and remain
//! reachable through raw dispatch; Phase 2 will generalise.

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorResult};
use crate::codegen::camel_to_snake;
use crate::parser::vk_types::{CommandParam, VulkanCommand, VulkanType};

/// Hand-written handle types that already have ergonomic `safe`
/// wrappers. Generation for these is suppressed to avoid name clashes
/// and to leave the existing ergonomic API unchallenged.
const HAND_WRITTEN: &[&str] = &[
    "VkInstance",
    "VkPhysicalDevice",
    "VkDevice",
    "VkQueue",
    "VkCommandBuffer",
    "VkCommandPool",
    "VkBuffer",
    "VkImage",
    "VkImageView",
    "VkSampler",
    "VkShaderModule",
    "VkPipeline",
    "VkPipelineLayout",
    "VkPipelineCache",
    "VkRenderPass",
    "VkFramebuffer",
    "VkSurfaceKHR",
    "VkSwapchainKHR",
    "VkDescriptorPool",
    "VkDescriptorSet",
    "VkDescriptorSetLayout",
    "VkDeviceMemory",
    "VkQueryPool",
    "VkFence",
    "VkSemaphore",
    "VkEvent",
    // Debug / surface extensions are tightly coupled to the hand-written
    // Instance and have bespoke setup paths; skip auto-wrap.
    "VkDebugUtilsMessengerEXT",
    "VkDebugReportCallbackEXT",
    // Display chain (VkDisplayKHR, VkDisplayModeKHR) is owned by the
    // physical device and has no destroy function — skip.
    "VkDisplayKHR",
    "VkDisplayModeKHR",
];

fn is_hand_written(name: &str) -> bool {
    HAND_WRITTEN.contains(&name)
}

/// Turn a Vulkan handle name (`VkAccelerationStructureKHR`) into the
/// Rust wrapper name (`AccelerationStructureKHR`) — just strip the
/// leading `Vk`.
fn wrapper_name(handle: &str) -> String {
    handle.strip_prefix("Vk").unwrap_or(handle).to_string()
}

/// Turn a Vulkan command name (`vkCreateAccelerationStructureKHR`)
/// into the snake-cased method name used in doc-comments (not for
/// actual dispatch — we call the raw fn pointer directly).
#[allow(dead_code)]
fn command_snake(cmd: &str) -> String {
    camel_to_snake(cmd.strip_prefix("vk").unwrap_or(cmd))
}

/// A handle type that qualifies for Phase-1 auto-wrap generation.
struct Candidate<'a> {
    handle: &'a VulkanType,
    create: &'a VulkanCommand,
    destroy: &'a VulkanCommand,
    /// Name of the Create command's CreateInfo parameter type, e.g.
    /// `VkAccelerationStructureCreateInfoKHR`.
    create_info_type: String,
}

/// Identify the Create/Allocate command that produces `handle`, if any.
/// The match is on name convention alone — `vkCreate<Name>` or
/// `vkAllocate<Name>`. We reject multi-handle variants (`vkCreate*s`)
/// and any command whose last parameter is not a `*mut VkHandle` with
/// the exact handle type we're looking for.
fn find_create_command<'a>(
    handle_name: &str,
    commands: &'a [VulkanCommand],
) -> Option<&'a VulkanCommand> {
    let stripped = handle_name.strip_prefix("Vk")?;
    let candidates: [String; 2] = [format!("vkCreate{}", stripped), format!("vkAllocate{}", stripped)];
    for cmd in commands {
        if cmd.is_alias {
            continue;
        }
        if !candidates.iter().any(|c| c == &cmd.name) {
            continue;
        }
        // Must have exactly four parameters:
        //   (parent_handle, const CreateInfo*, const Allocator*, VkHandle* out)
        if cmd.parameters.len() != 4 {
            continue;
        }
        if !is_single_handle_out(&cmd.parameters[3], handle_name) {
            continue;
        }
        if !is_create_info_pointer(&cmd.parameters[1]) {
            continue;
        }
        if !is_allocator_pointer(&cmd.parameters[2]) {
            continue;
        }
        return Some(cmd);
    }
    None
}

fn find_destroy_command<'a>(
    handle_name: &str,
    commands: &'a [VulkanCommand],
) -> Option<&'a VulkanCommand> {
    let stripped = handle_name.strip_prefix("Vk")?;
    let candidates: [String; 2] = [format!("vkDestroy{}", stripped), format!("vkFree{}", stripped)];
    for cmd in commands {
        if cmd.is_alias {
            continue;
        }
        if !candidates.iter().any(|c| c == &cmd.name) {
            continue;
        }
        // Must have exactly three parameters:
        //   (parent_handle, VkHandle, const Allocator*)
        if cmd.parameters.len() != 3 {
            continue;
        }
        if cmd.parameters[1].type_name != handle_name
            || param_pointer_level(&cmd.parameters[1]) != 0
        {
            continue;
        }
        if !is_allocator_pointer(&cmd.parameters[2]) {
            continue;
        }
        return Some(cmd);
    }
    None
}

fn param_pointer_level(p: &CommandParam) -> usize {
    p.definition.matches('*').count()
}

fn is_single_handle_out(p: &CommandParam, handle_name: &str) -> bool {
    p.type_name == handle_name
        && param_pointer_level(p) == 1
        && p.len.is_none()
        && p.definition.contains('*')
        && !p.definition.contains("const")
}

fn is_create_info_pointer(p: &CommandParam) -> bool {
    p.type_name.contains("CreateInfo")
        && param_pointer_level(p) == 1
        && p.definition.contains("const")
}

fn is_allocator_pointer(p: &CommandParam) -> bool {
    p.type_name == "VkAllocationCallbacks" && param_pointer_level(p) == 1
}

fn emit_wrapper(cand: &Candidate) -> String {
    let handle_vk = cand.handle.name.as_str();
    let wrapper = wrapper_name(handle_vk);
    let create_fp = cand.create.name.as_str(); // dispatch table field
    let destroy_fp = cand.destroy.name.as_str();
    let parent_handle = cand.create.parameters[0].type_name.as_str();
    let create_info_ty = cand.create_info_type.as_str();

    // For Phase 1 we only support Device-parented handles (the vast
    // majority). Instance/PhysicalDevice-parented handles are skipped.
    debug_assert_eq!(parent_handle, "VkDevice");

    // Dispatchable handles are `*mut c_void` and initialise to
    // `std::ptr::null_mut()`; non-dispatchable handles are `u64` and
    // initialise to `0`. Detect via the macro used in vk.xml.
    let is_dispatchable = cand
        .handle
        .raw_content
        .contains("VK_DEFINE_HANDLE(");
    let zero_init = if is_dispatchable {
        "std::ptr::null_mut()"
    } else {
        "0"
    };

    let mut out = String::new();
    out.push_str(&format!(
        "/// Safe RAII wrapper for [`{handle_vk}`](crate::raw::bindings::{handle_vk}).\n\
         ///\n\
         /// Generated from the `{create_fp}` / `{destroy_fp}` command pair.\n\
         /// The wrapped handle is destroyed automatically on drop.\n\
         pub struct {wrapper} {{\n\
         \x20   handle: crate::raw::bindings::{handle_vk},\n\
         \x20   device: std::sync::Arc<crate::safe::device::DeviceInner>,\n\
         }}\n\n"
    ));

    out.push_str(&format!(
        "impl {wrapper} {{\n\
         \x20   /// Create a new `{wrapper}` from a filled-in `{create_info_ty}`.\n\
         \x20   ///\n\
         \x20   /// Returns `Error::MissingFunction` if the owning extension\n\
         \x20   /// wasn't enabled on the `Device`.\n\
         \x20   pub fn create(\n\
         \x20       device: &crate::safe::Device,\n\
         \x20       create_info: &crate::raw::bindings::{create_info_ty},\n\
         \x20   ) -> crate::safe::Result<Self> {{\n\
         \x20       let dispatch_fn = device\n\
         \x20           .inner\n\
         \x20           .dispatch\n\
         \x20           .{create_fp}\n\
         \x20           .ok_or(crate::safe::Error::MissingFunction(\"{create_fp}\"))?;\n\
         \x20       let mut handle: crate::raw::bindings::{handle_vk} = {zero_init};\n\
         \x20       // Safety: handle is valid; create_info outlives the call.\n\
         \x20       crate::safe::check(unsafe {{\n\
         \x20           dispatch_fn(\n\
         \x20               device.inner.handle,\n\
         \x20               create_info,\n\
         \x20               std::ptr::null(),\n\
         \x20               &mut handle,\n\
         \x20           )\n\
         \x20       }})?;\n\
         \x20       Ok(Self {{ handle, device: std::sync::Arc::clone(&device.inner) }})\n\
         \x20   }}\n\n\
         \x20   /// Raw `{handle_vk}` for use with raw dispatch.\n\
         \x20   pub fn raw(&self) -> crate::raw::bindings::{handle_vk} {{\n\
         \x20       self.handle\n\
         \x20   }}\n\
         }}\n\n"
    ));

    out.push_str(&format!(
        "impl Drop for {wrapper} {{\n\
         \x20   fn drop(&mut self) {{\n\
         \x20       if let Some(destroy) = self.device.dispatch.{destroy_fp} {{\n\
         \x20           // Safety: we're the sole owner and the parent device\n\
         \x20           // is kept alive by the `Arc`.\n\
         \x20           unsafe {{ destroy(self.device.handle, self.handle, std::ptr::null()) }};\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n\n"
    ));

    // SAFETY: wrapped handles are `Send + Sync` at the Rust level.
    // Non-dispatchable Vulkan handles are `u64` under the hood and
    // have no interior mutability. Actual thread-safety requirements
    // (external sync) still apply when making Vulkan calls — the
    // trait impls here only express that moving the wrapper across
    // threads is sound.
    out.push_str(&format!("unsafe impl Send for {wrapper} {{}}\n"));
    out.push_str(&format!("unsafe impl Sync for {wrapper} {{}}\n\n"));

    out
}

pub struct SafeHandlesStats {
    pub generated: usize,
    pub skipped: usize,
}

pub fn generate_safe_handles(
    intermediate_dir: &Path,
    output_path: &Path,
) -> GeneratorResult<SafeHandlesStats> {
    // Load types.json (handles).
    let types_path = intermediate_dir.join("types.json");
    let content = fs::read_to_string(&types_path).map_err(GeneratorError::Io)?;
    let types: Vec<VulkanType> = serde_json::from_str(&content)?;
    let handles: Vec<&VulkanType> = types
        .iter()
        .filter(|t| t.category == "handle" && !t.is_alias && t.deprecated.is_none())
        .collect();

    // Load functions.json (commands).
    let fns_path = intermediate_dir.join("functions.json");
    let content = fs::read_to_string(&fns_path).map_err(GeneratorError::Io)?;
    let commands: Vec<VulkanCommand> = serde_json::from_str(&content)?;

    let mut skipped = 0usize;
    let mut candidates: Vec<Candidate> = Vec::new();

    for handle in &handles {
        if is_hand_written(&handle.name) {
            skipped += 1;
            continue;
        }
        let Some(create) = find_create_command(&handle.name, &commands) else {
            skipped += 1;
            continue;
        };
        let Some(destroy) = find_destroy_command(&handle.name, &commands) else {
            skipped += 1;
            continue;
        };
        // Only Device-parented for Phase 1 (by far the most common case).
        if create.parameters[0].type_name != "VkDevice" {
            skipped += 1;
            continue;
        }
        candidates.push(Candidate {
            handle,
            create,
            destroy,
            create_info_type: create.parameters[1].type_name.clone(),
        });
    }

    // Stable emission order so the generated file is deterministic.
    candidates.sort_by(|a, b| a.handle.name.cmp(&b.handle.name));

    let mut file = String::new();
    file.push_str(
        "// Generated by vulkan_gen::safe_handles_gen — do not edit.\n\
         //\n\
         // Phase-1 auto-safe-layer: RAII wrappers for every Vulkan handle\n\
         // type whose Create / Destroy pair fits the simple four-/three-\n\
         // parameter shape. Included from `vulkane/src/safe/auto.rs`.\n\n",
    );
    for cand in &candidates {
        file.push_str(&emit_wrapper(cand));
    }

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(GeneratorError::Io)?;
    }
    fs::write(output_path, file).map_err(GeneratorError::Io)?;

    Ok(SafeHandlesStats {
        generated: candidates.len(),
        skipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_param(name: &str, type_name: &str, definition: &str) -> CommandParam {
        CommandParam {
            name: name.to_string(),
            type_name: type_name.to_string(),
            optional: None,
            len: None,
            altlen: None,
            externsync: None,
            noautovalidity: None,
            objecttype: None,
            stride: None,
            validstructs: None,
            api: None,
            deprecated: None,
            comment: None,
            definition: definition.to_string(),
            raw_content: String::new(),
            source_line: None,
        }
    }

    fn mk_cmd(name: &str, params: Vec<CommandParam>) -> VulkanCommand {
        VulkanCommand {
            name: name.to_string(),
            return_type: "VkResult".to_string(),
            comment: None,
            successcodes: None,
            errorcodes: None,
            alias: None,
            api: None,
            deprecated: None,
            cmdbufferlevel: None,
            pipeline: None,
            queues: None,
            renderpass: None,
            videocoding: None,
            parameters: params,
            raw_content: String::new(),
            is_alias: false,
            source_line: None,
        }
    }

    fn mk_handle(name: &str) -> VulkanType {
        VulkanType {
            name: name.to_string(),
            category: "handle".to_string(),
            definition: None,
            api: None,
            requires: None,
            bitvalues: None,
            parent: Some("VkDevice".to_string()),
            objtypeenum: None,
            alias: None,
            deprecated: None,
            comment: None,
            raw_content: String::new(),
            type_references: Vec::new(),
            is_alias: false,
        }
    }

    #[test]
    fn skips_hand_written_handles() {
        assert!(is_hand_written("VkDevice"));
        assert!(is_hand_written("VkBuffer"));
        assert!(!is_hand_written("VkAccelerationStructureKHR"));
    }

    #[test]
    fn find_create_matches_four_param_shape() {
        let handle = mk_handle("VkFoo");
        let commands = vec![mk_cmd(
            "vkCreateFoo",
            vec![
                mk_param("device", "VkDevice", "VkDevice device"),
                mk_param("pCreateInfo", "VkFooCreateInfo", "const VkFooCreateInfo* pCreateInfo"),
                mk_param("pAllocator", "VkAllocationCallbacks", "const VkAllocationCallbacks* pAllocator"),
                mk_param("pHandle", "VkFoo", "VkFoo* pHandle"),
            ],
        )];
        let found = find_create_command(&handle.name, &commands);
        assert!(found.is_some());
    }

    #[test]
    fn find_create_rejects_multi_handle_out() {
        // vkCreateGraphicsPipelines-style: array of handles out, not one.
        let handle = mk_handle("VkPipeline");
        let mut p = mk_param("pPipelines", "VkPipeline", "VkPipeline* pPipelines");
        p.len = Some("createInfoCount".to_string());
        let commands = vec![mk_cmd(
            "vkCreatePipeline",
            vec![
                mk_param("device", "VkDevice", "VkDevice device"),
                mk_param("pCreateInfo", "VkPipelineCreateInfo", "const VkPipelineCreateInfo* p"),
                mk_param("pAllocator", "VkAllocationCallbacks", "const VkAllocationCallbacks* a"),
                p,
            ],
        )];
        // len=createInfoCount on the output => multi-handle, should be rejected.
        assert!(find_create_command(&handle.name, &commands).is_none());
    }

    #[test]
    fn emitted_wrapper_contains_expected_items() {
        let handle = mk_handle("VkFooKHR");
        let create = mk_cmd(
            "vkCreateFooKHR",
            vec![
                mk_param("device", "VkDevice", "VkDevice device"),
                mk_param("pCreateInfo", "VkFooCreateInfoKHR", "const VkFooCreateInfoKHR* pCreateInfo"),
                mk_param("pAllocator", "VkAllocationCallbacks", "const VkAllocationCallbacks* pAllocator"),
                mk_param("pFoo", "VkFooKHR", "VkFooKHR* pFoo"),
            ],
        );
        let destroy = mk_cmd(
            "vkDestroyFooKHR",
            vec![
                mk_param("device", "VkDevice", "VkDevice device"),
                mk_param("foo", "VkFooKHR", "VkFooKHR foo"),
                mk_param("pAllocator", "VkAllocationCallbacks", "const VkAllocationCallbacks* pAllocator"),
            ],
        );
        let cand = Candidate {
            handle: &handle,
            create: &create,
            destroy: &destroy,
            create_info_type: "VkFooCreateInfoKHR".to_string(),
        };
        let code = emit_wrapper(&cand);
        assert!(code.contains("pub struct FooKHR"));
        assert!(code.contains("impl FooKHR"));
        assert!(code.contains("vkCreateFooKHR"));
        assert!(code.contains("vkDestroyFooKHR"));
        assert!(code.contains("impl Drop for FooKHR"));
        assert!(code.contains("unsafe impl Send for FooKHR"));
        assert!(code.contains("unsafe impl Sync for FooKHR"));
    }

}
