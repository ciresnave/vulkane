//! Generator for Phase 2 of the auto-safe layer: one safe wrapper
//! method per Vulkan command, grouped into ext traits by the command's
//! dispatch target.
//!
//! Emits five files:
//!
//! - `auto_device_ext_generated.rs` — `DeviceExt` (first arg is `VkDevice`)
//! - `auto_instance_ext_generated.rs` — `InstanceExt` (first arg is `VkInstance`)
//! - `auto_physical_device_ext_generated.rs` — `PhysicalDeviceExt` (first arg is `VkPhysicalDevice`)
//! - `auto_queue_ext_generated.rs` — `QueueExt` (first arg is `VkQueue`)
//! - `auto_command_buffer_ext_generated.rs` — `CommandBufferRecordingExt` (first arg is `VkCommandBuffer`)
//!
//! All included from `vulkane/src/safe/auto.rs`. Users opt in by
//! `use vulkane::safe::{DeviceExt, CommandBufferRecordingExt, …};`.
//!
//! # Emission philosophy
//!
//! Pragmatic. Cover every command with *some* safe method, even if not
//! ergonomic. Pattern-match a few shapes (slice + count, single output,
//! enumerate) and for anything else fall through to a pass-through
//! wrapper that takes raw Vulkan types and `unsafe {}`s around the
//! dispatch call internally so user code stays safe.
//!
//! # Skipped commands
//!
//! Commands that fit the RAII Create/Destroy pattern (and are therefore
//! handled by `safe_handles_gen`) are skipped here, as are `vkDestroy*`
//! and `vkFree<Pool>s` helpers and any command that takes a non-dispatch
//! first parameter the generator can't classify.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorResult};
use crate::codegen::camel_to_snake;
use crate::parser::vk_types::{CommandParam, VulkanCommand};

pub struct SafeCommandsStats {
    pub device_methods: usize,
    pub instance_methods: usize,
    pub physical_device_methods: usize,
    pub queue_methods: usize,
    pub command_buffer_methods: usize,
    pub skipped: usize,
}

/// Dispatch target for a command — determines which ext trait it lands
/// on and how the generated body reaches the raw handle / dispatch
/// table.
#[derive(Copy, Clone)]
enum Target {
    Device,
    Instance,
    PhysicalDevice,
    Queue,
    CommandBuffer,
}

impl Target {
    fn from_first_param(ty: &str) -> Option<Self> {
        match ty {
            "VkDevice" => Some(Target::Device),
            "VkInstance" => Some(Target::Instance),
            "VkPhysicalDevice" => Some(Target::PhysicalDevice),
            "VkQueue" => Some(Target::Queue),
            "VkCommandBuffer" => Some(Target::CommandBuffer),
            _ => None,
        }
    }

    fn trait_name(self) -> &'static str {
        match self {
            Target::Device => "DeviceExt",
            Target::Instance => "InstanceExt",
            Target::PhysicalDevice => "PhysicalDeviceExt",
            Target::Queue => "QueueExt",
            Target::CommandBuffer => "CommandBufferRecordingExt",
        }
    }

    fn impl_target(self) -> &'static str {
        match self {
            Target::Device => "crate::safe::Device",
            Target::Instance => "crate::safe::Instance",
            Target::PhysicalDevice => "crate::safe::PhysicalDevice",
            Target::Queue => "crate::safe::Queue",
            Target::CommandBuffer => "crate::safe::CommandBufferRecording<'_>",
        }
    }

    /// Expression yielding `self`'s raw handle of the Vulkan type the
    /// command expects as its first parameter.
    fn raw_handle_expr(self) -> &'static str {
        match self {
            Target::Device => "self.inner.handle",
            Target::Instance => "self.inner.handle",
            Target::PhysicalDevice => "self.handle",
            Target::Queue => "self.handle",
            Target::CommandBuffer => "self.raw_cmd()",
        }
    }

    /// Expression yielding the dispatch table reference.
    fn dispatch_expr(self) -> &'static str {
        match self {
            Target::Device => "self.inner.dispatch",
            Target::Instance => "self.inner.dispatch",
            Target::PhysicalDevice => "self.instance.dispatch",
            Target::Queue => "self.device.dispatch",
            Target::CommandBuffer => "self.device_dispatch()",
        }
    }

    /// Whether methods on this target take `&self` (default) or
    /// `&mut self`. Command-buffer recording state mutates, so those
    /// methods are `&mut self`.
    fn self_kind(self) -> &'static str {
        match self {
            Target::CommandBuffer => "&mut self",
            _ => "&self",
        }
    }
}

/// Commands already covered by `safe_handles_gen` (generate/allocate
/// pairs) — suppress here so we don't emit two safe paths for the same
/// call.
fn is_phase1_handled_command(name: &str) -> bool {
    // Suppress every `vkCreate*`, `vkAllocate*` (handled as RAII constructors)
    // and every `vkDestroy*` / `vkFree*` (Drop impls). These cover several
    // commands that aren't actually captured by Phase 1's RAII generator
    // (e.g. pool-allocate/free), but users don't need safe wrappers for
    // those either — they're covered by the hand-written `CommandBuffer` /
    // `DescriptorSet` / etc. wrappers.
    name.starts_with("vkCreate")
        || name.starts_with("vkDestroy")
        || name.starts_with("vkAllocate")
        || name.starts_with("vkFree")
}

/// Rust identifier name for a method, derived from the Vulkan command
/// name. Preserves the `vk_` prefix so generated methods are easy to
/// distinguish from hand-written ergonomic wrappers.
fn method_name(cmd_name: &str) -> String {
    // "vkCmdTraceRaysKHR" -> "vk_cmd_trace_rays_khr"
    camel_to_snake(cmd_name)
}

/// Number of `*` in a parameter definition — i.e. its pointer depth.
fn pointer_level(p: &CommandParam) -> usize {
    p.definition.matches('*').count()
}

fn is_allocator_callbacks(p: &CommandParam) -> bool {
    p.type_name == "VkAllocationCallbacks" && pointer_level(p) == 1
}

/// Map a parameter type to its Rust spelling, matching the pointer
/// convention used by `function_gen` (which emits the raw fn-pointer
/// signatures the dispatch table holds). Innermost pointer wears any
/// `const` qualifier; outer pointers default to `*mut` — that's the
/// convention vulkane's raw bindings use, so a pass-through safe
/// method must adopt the same shape for the call site to type-check.
///
/// `c_void` / `c_char` are emitted fully qualified via `core::ffi` so
/// we don't need `use` statements inside the generated include-file.
fn qualified_raw_type(p: &CommandParam) -> String {
    let base = match p.type_name.as_str() {
        "void" => "core::ffi::c_void".to_string(),
        "char" => "core::ffi::c_char".to_string(),
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
        other => format!("crate::raw::bindings::{}", other),
    };

    let depth = pointer_level(p);
    if depth == 0 {
        return base;
    }
    let const_qualified = p.definition.contains("const");

    // Innermost pointer gets `const` if the definition is const-qualified;
    // outer pointers are all `*mut`. Mirrors function_gen's emission.
    let mut out = base;
    for i in 0..depth {
        if i == 0 && const_qualified {
            out = format!("*const {out}");
        } else {
            out = format!("*mut {out}");
        }
    }
    out
}

/// Escape a parameter name if it collides with a Rust keyword.
fn escape_param_name(name: &str) -> String {
    match name {
        "type" | "match" | "impl" | "fn" | "let" | "mut" | "const" | "static" | "if" | "else"
        | "while" | "for" | "loop" | "break" | "continue" | "return" | "struct" | "enum"
        | "trait" | "mod" | "pub" | "use" | "extern" | "crate" | "self" | "Self" | "super"
        | "where" | "async" | "await" | "dyn" | "abstract" | "become" | "box" | "do" | "final"
        | "macro" | "override" | "priv" | "typeof" | "unsized" | "virtual" | "yield" | "try"
        | "union" | "ref" => format!("r#{}", name),
        _ => name.to_string(),
    }
}

fn returns_vk_result(cmd: &VulkanCommand) -> bool {
    cmd.return_type == "VkResult"
}

fn returns_void(cmd: &VulkanCommand) -> bool {
    cmd.return_type == "void"
}

/// Emit one generated method + impl-body string for `cmd`, plus its
/// matching trait-signature string. Returns `None` if the command is
/// skipped.
fn emit_method(cmd: &VulkanCommand, target: Target) -> Option<(String, String)> {
    let method = method_name(&cmd.name);

    // First parameter is always the dispatch-target handle — we drop
    // it from the signature and pass `self`'s raw handle in the body.
    if cmd.parameters.is_empty() {
        return None;
    }

    // Build the signature parameters. We pass every other parameter
    // through as a raw value/pointer of its exact Vulkan type — no
    // slice detection, no Option<&T> conversion. This is the "works
    // for every command without clever pattern-matching" baseline; the
    // caller builds whatever structure they need and hands us raw
    // pointers. Safer than unsafe dispatch; less ergonomic than a
    // hand-written wrapper.
    let mut sig_params: Vec<String> = Vec::new();
    let mut call_args: Vec<String> = Vec::new();
    call_args.push(target.raw_handle_expr().to_string()); // first arg: self's raw handle

    for (i, p) in cmd.parameters.iter().enumerate() {
        if i == 0 {
            continue; // already pushed via raw_handle_expr()
        }
        if is_allocator_callbacks(p) {
            // Always pass null — we don't expose allocation callbacks.
            call_args.push("std::ptr::null()".to_string());
            continue;
        }
        let rust_ty = qualified_raw_type(p);
        let name = escape_param_name(&p.name);
        sig_params.push(format!("{name}: {rust_ty}"));
        call_args.push(name);
    }

    let self_kind = target.self_kind();
    let params_str = if sig_params.is_empty() {
        String::new()
    } else {
        format!(", {}", sig_params.join(", "))
    };

    let return_type = if returns_vk_result(cmd) {
        "crate::safe::Result<crate::raw::bindings::VkResult>"
    } else if returns_void(cmd) {
        "()"
    } else {
        // Non-void non-VkResult return (e.g. u32 for some query APIs).
        // Emit as the raw return type.
        match cmd.return_type.as_str() {
            "uint32_t" => "u32",
            "uint64_t" => "u64",
            "int32_t" => "i32",
            "int64_t" => "i64",
            "float" => "f32",
            "double" => "f64",
            "size_t" => "usize",
            other if other.starts_with("Vk") => {
                // Typed handles, enums, bitmasks — pass through.
                // We need to emit fully qualified.
                return Some(emit_passthrough_typed_return(cmd, target, other));
            }
            _ => return None, // Skip truly weird return types.
        }
    };

    let signature = format!("    fn {method}({self_kind}{params_str}) -> {return_type};\n");

    let dispatch_expr = target.dispatch_expr();
    let fp_name = &cmd.name;
    let call = format!("{}({})", "f", call_args.join(", "));

    let mut body = String::new();
    body.push_str(&format!(
        "    fn {method}({self_kind}{params_str}) -> {return_type} {{\n"
    ));
    body.push_str(&format!(
        "        let f = {dispatch_expr}.{fp_name}.expect(\"{fp_name} not loaded — did you enable its extension?\");\n"
    ));

    if returns_vk_result(cmd) {
        // Return Result<VkResult> so callers see VK_SUCCESS / VK_INCOMPLETE /
        // VK_SUBOPTIMAL_KHR / etc. distinctly. Any *error* code is
        // translated to Err via `check`-like logic.
        body.push_str(&format!("        let r = unsafe {{ {call} }};\n"));
        body.push_str(
            "        if (r as i32) < 0 { Err(crate::safe::Error::Vk(r)) } else { Ok(r) }\n",
        );
    } else if returns_void(cmd) {
        body.push_str(&format!("        unsafe {{ {call} }};\n"));
    } else {
        body.push_str(&format!("        unsafe {{ {call} }}\n"));
    }
    body.push_str("    }\n");
    Some((signature, body))
}

/// Emit a pass-through method for commands whose return type is a
/// typed Vulkan value (enum, handle, bitmask). Body just calls the
/// function pointer and returns the result.
fn emit_passthrough_typed_return(
    cmd: &VulkanCommand,
    target: Target,
    return_ty: &str,
) -> (String, String) {
    let method = method_name(&cmd.name);
    let mut sig_params: Vec<String> = Vec::new();
    let mut call_args: Vec<String> = Vec::new();
    call_args.push(target.raw_handle_expr().to_string());

    for (i, p) in cmd.parameters.iter().enumerate() {
        if i == 0 {
            continue;
        }
        if is_allocator_callbacks(p) {
            call_args.push("std::ptr::null()".to_string());
            continue;
        }
        let rust_ty = qualified_raw_type(p);
        let name = escape_param_name(&p.name);
        sig_params.push(format!("{name}: {rust_ty}"));
        call_args.push(name);
    }

    let self_kind = target.self_kind();
    let params_str = if sig_params.is_empty() {
        String::new()
    } else {
        format!(", {}", sig_params.join(", "))
    };

    let return_type_str = format!("crate::raw::bindings::{}", return_ty);
    let signature = format!("    fn {method}({self_kind}{params_str}) -> {return_type_str};\n");

    let dispatch_expr = target.dispatch_expr();
    let fp_name = &cmd.name;
    let call = format!("{}({})", "f", call_args.join(", "));

    let mut body = String::new();
    body.push_str(&format!(
        "    fn {method}({self_kind}{params_str}) -> {return_type_str} {{\n"
    ));
    body.push_str(&format!(
        "        let f = {dispatch_expr}.{fp_name}.expect(\"{fp_name} not loaded — did you enable its extension?\");\n"
    ));
    body.push_str(&format!("        unsafe {{ {call} }}\n"));
    body.push_str("    }\n");
    (signature, body)
}

/// Deduplicate Vulkan commands by stripping aliases and any repeats.
fn unique_commands(commands: &[VulkanCommand]) -> Vec<&VulkanCommand> {
    let mut seen: HashSet<&str> = HashSet::new();
    commands
        .iter()
        .filter(|c| !c.is_alias && c.deprecated.is_none() && seen.insert(c.name.as_str()))
        .collect()
}

fn emit_file(
    target: Target,
    cmds: &[&VulkanCommand],
) -> (String, usize) {
    let trait_name = target.trait_name();
    let impl_target = target.impl_target();

    let mut out = String::new();
    out.push_str(&format!(
        "// Generated by vulkan_gen::safe_commands_gen — do not edit.\n\
         //\n\
         // Phase-2 auto-safe-layer: one method per Vulkan command whose\n\
         // first argument is the `{target_ty}` handle this trait extends.\n\
         // Every method takes raw Vulkan parameter types — users build\n\
         // the CreateInfo / query struct and hand us raw pointers; we\n\
         // wrap the call in `unsafe {{}}` so user code stays safe.\n\n",
        target_ty = match target {
            Target::Device => "VkDevice",
            Target::Instance => "VkInstance",
            Target::PhysicalDevice => "VkPhysicalDevice",
            Target::Queue => "VkQueue",
            Target::CommandBuffer => "VkCommandBuffer",
        }
    ));

    // Trait header.
    //
    // Allow-list covers every clippy lint the pass-through shape will
    // inevitably trip: raw-pointer derefs (we're wrapping C calls),
    // trailing `-> ()` on void commands (consistent shape for the
    // emitter), and the occasional redundant auto-deref inside
    // dispatch accessors.
    out.push_str(&format!(
        "#[allow(non_snake_case, clippy::too_many_arguments, clippy::missing_safety_doc, clippy::not_unsafe_ptr_arg_deref, clippy::unused_unit, clippy::explicit_auto_deref)]\n\
         pub trait {trait_name} {{\n"
    ));

    let mut trait_body = String::new();
    let mut impl_body = String::new();
    let mut emitted = 0usize;

    for cmd in cmds {
        if let Some((sig, body)) = emit_method(cmd, target) {
            trait_body.push_str(&sig);
            impl_body.push_str(&body);
            emitted += 1;
        }
    }

    out.push_str(&trait_body);
    out.push_str("}\n\n");

    out.push_str(&format!(
        "#[allow(non_snake_case, clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref, clippy::unused_unit, clippy::explicit_auto_deref)]\n\
         impl {trait_name} for {impl_target} {{\n"
    ));
    out.push_str(&impl_body);
    out.push_str("}\n");

    (out, emitted)
}

pub fn generate_safe_commands(
    intermediate_dir: &Path,
    output_dir: &Path,
) -> GeneratorResult<SafeCommandsStats> {
    let fns_path = intermediate_dir.join("functions.json");
    let content = fs::read_to_string(&fns_path).map_err(GeneratorError::Io)?;
    let commands: Vec<VulkanCommand> = serde_json::from_str(&content)?;

    let uniq = unique_commands(&commands);

    let mut by_target: std::collections::HashMap<&'static str, Vec<&VulkanCommand>> =
        std::collections::HashMap::new();
    let mut skipped = 0usize;

    for cmd in &uniq {
        if is_phase1_handled_command(&cmd.name) {
            skipped += 1;
            continue;
        }
        if cmd.parameters.is_empty() {
            skipped += 1;
            continue;
        }
        let target = match Target::from_first_param(&cmd.parameters[0].type_name) {
            Some(t) => t,
            None => {
                skipped += 1;
                continue;
            }
        };
        by_target
            .entry(target.trait_name())
            .or_default()
            .push(cmd);
    }

    fs::create_dir_all(output_dir).map_err(GeneratorError::Io)?;

    let mut stats = SafeCommandsStats {
        device_methods: 0,
        instance_methods: 0,
        physical_device_methods: 0,
        queue_methods: 0,
        command_buffer_methods: 0,
        skipped,
    };

    for target in [
        Target::Device,
        Target::Instance,
        Target::PhysicalDevice,
        Target::Queue,
        Target::CommandBuffer,
    ] {
        let cmds: Vec<&VulkanCommand> = by_target
            .get(target.trait_name())
            .cloned()
            .unwrap_or_default();

        let (code, emitted) = emit_file(target, &cmds);

        // Any cmds the emitter dropped also count as skipped.
        let dropped = cmds.len().saturating_sub(emitted);
        stats.skipped += dropped;

        match target {
            Target::Device => stats.device_methods = emitted,
            Target::Instance => stats.instance_methods = emitted,
            Target::PhysicalDevice => stats.physical_device_methods = emitted,
            Target::Queue => stats.queue_methods = emitted,
            Target::CommandBuffer => stats.command_buffer_methods = emitted,
        }

        let filename = match target {
            Target::Device => "auto_device_ext_generated.rs",
            Target::Instance => "auto_instance_ext_generated.rs",
            Target::PhysicalDevice => "auto_physical_device_ext_generated.rs",
            Target::Queue => "auto_queue_ext_generated.rs",
            Target::CommandBuffer => "auto_command_buffer_ext_generated.rs",
        };
        fs::write(output_dir.join(filename), code).map_err(GeneratorError::Io)?;
    }

    Ok(stats)
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

    fn mk_cmd(name: &str, ret: &str, params: Vec<CommandParam>) -> VulkanCommand {
        VulkanCommand {
            name: name.to_string(),
            return_type: ret.to_string(),
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

    #[test]
    fn method_name_keeps_vk_prefix() {
        assert_eq!(method_name("vkCmdTraceRaysKHR"), "vk_cmd_trace_rays_khr");
        assert_eq!(method_name("vkQueueSubmit"), "vk_queue_submit");
    }

    #[test]
    fn target_routing() {
        assert!(matches!(
            Target::from_first_param("VkDevice"),
            Some(Target::Device)
        ));
        assert!(matches!(
            Target::from_first_param("VkCommandBuffer"),
            Some(Target::CommandBuffer)
        ));
        assert!(Target::from_first_param("VkBuffer").is_none());
    }

    #[test]
    fn create_destroy_are_skipped() {
        assert!(is_phase1_handled_command("vkCreateAccelerationStructureKHR"));
        assert!(is_phase1_handled_command("vkDestroyBuffer"));
        assert!(is_phase1_handled_command("vkAllocateCommandBuffers"));
        assert!(is_phase1_handled_command("vkFreeDescriptorSets"));
        assert!(!is_phase1_handled_command("vkCmdDraw"));
        assert!(!is_phase1_handled_command("vkGetPhysicalDeviceProperties2"));
    }

    #[test]
    fn void_cmd_emits_call_without_result() {
        let cmd = mk_cmd(
            "vkCmdDraw",
            "void",
            vec![
                mk_param("commandBuffer", "VkCommandBuffer", "VkCommandBuffer commandBuffer"),
                mk_param("vertexCount", "uint32_t", "uint32_t vertexCount"),
                mk_param("instanceCount", "uint32_t", "uint32_t instanceCount"),
                mk_param("firstVertex", "uint32_t", "uint32_t firstVertex"),
                mk_param("firstInstance", "uint32_t", "uint32_t firstInstance"),
            ],
        );
        let (sig, body) = emit_method(&cmd, Target::CommandBuffer).expect("emitted");
        assert!(sig.contains("fn vk_cmd_draw(&mut self"));
        assert!(sig.contains("vertexCount: u32"));
        assert!(sig.contains("-> ()"));
        assert!(body.contains("self.raw_cmd()"));
        assert!(body.contains("vkCmdDraw"));
        assert!(body.contains("unsafe"));
    }

    #[test]
    fn vk_result_cmd_returns_result() {
        let cmd = mk_cmd(
            "vkDeviceWaitIdle",
            "VkResult",
            vec![mk_param("device", "VkDevice", "VkDevice device")],
        );
        let (sig, body) = emit_method(&cmd, Target::Device).expect("emitted");
        assert!(sig.contains("fn vk_device_wait_idle(&self)"));
        assert!(sig.contains("crate::safe::Result<crate::raw::bindings::VkResult>"));
        assert!(body.contains("Err(crate::safe::Error::Vk(r))"));
        assert!(body.contains("Ok(r)"));
    }

    #[test]
    fn allocator_callbacks_are_always_null() {
        let cmd = mk_cmd(
            "vkDestroySomething",
            "void",
            vec![
                mk_param("device", "VkDevice", "VkDevice device"),
                mk_param("thing", "VkSomething", "VkSomething thing"),
                mk_param("pAllocator", "VkAllocationCallbacks", "const VkAllocationCallbacks* pAllocator"),
            ],
        );
        let (sig, body) = emit_method(&cmd, Target::Device).expect("emitted");
        assert!(!sig.contains("pAllocator"), "pAllocator hidden from signature");
        assert!(body.contains("std::ptr::null()"), "pAllocator passed as null");
    }
}
