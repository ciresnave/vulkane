//! Safe wrapper around `VK_EXT_debug_utils` — validation layer messages.
//!
//! When the `VK_EXT_debug_utils` instance extension is enabled (typically
//! together with the `VK_LAYER_KHRONOS_validation` layer), the Vulkan
//! implementation forwards validation, performance, and general messages to
//! a callback you register on the instance. This module wraps that callback
//! plumbing so you can supply a plain Rust closure.
//!
//! The recommended way to enable validation in spock is the
//! [`InstanceCreateInfo::validation`](super::InstanceCreateInfo::validation)
//! convenience: it auto-enables the layer and the extension and installs a
//! default `eprintln!`-style callback. For finer control, set
//! [`enabled_layers`](super::InstanceCreateInfo::enabled_layers),
//! [`enabled_extensions`](super::InstanceCreateInfo::enabled_extensions), and
//! [`debug_callback`](super::InstanceCreateInfo::debug_callback) explicitly.

use crate::raw::bindings::*;
use std::ffi::CStr;

/// Severity bit of a debug message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DebugMessageSeverity(pub u32);

impl DebugMessageSeverity {
    pub const VERBOSE: Self = Self(0x1);
    pub const INFO: Self = Self(0x10);
    pub const WARNING: Self = Self(0x100);
    pub const ERROR: Self = Self(0x1000);

    pub const ALL: Self = Self(Self::VERBOSE.0 | Self::INFO.0 | Self::WARNING.0 | Self::ERROR.0);

    /// All severity bits at WARNING and above.
    pub const WARNING_AND_ABOVE: Self = Self(Self::WARNING.0 | Self::ERROR.0);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Single-bit display label, for human consumption.
    pub fn label(self) -> &'static str {
        if self.contains(Self::ERROR) {
            "ERROR"
        } else if self.contains(Self::WARNING) {
            "WARN"
        } else if self.contains(Self::INFO) {
            "INFO"
        } else if self.contains(Self::VERBOSE) {
            "VERBOSE"
        } else {
            "?"
        }
    }
}

impl std::ops::BitOr for DebugMessageSeverity {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Type bits of a debug message — what subsystem produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DebugMessageType(pub u32);

impl DebugMessageType {
    pub const GENERAL: Self = Self(0x1);
    pub const VALIDATION: Self = Self(0x2);
    pub const PERFORMANCE: Self = Self(0x4);
    pub const DEVICE_ADDRESS_BINDING: Self = Self(0x8);

    pub const ALL: Self = Self(
        Self::GENERAL.0 | Self::VALIDATION.0 | Self::PERFORMANCE.0 | Self::DEVICE_ADDRESS_BINDING.0,
    );

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Compact display label, for human consumption.
    pub fn label(self) -> &'static str {
        if self.contains(Self::VALIDATION) {
            "VALID"
        } else if self.contains(Self::PERFORMANCE) {
            "PERF"
        } else if self.contains(Self::GENERAL) {
            "GEN"
        } else if self.contains(Self::DEVICE_ADDRESS_BINDING) {
            "ADDR"
        } else {
            "?"
        }
    }
}

impl std::ops::BitOr for DebugMessageType {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// One debug message delivered to a [`DebugCallback`].
///
/// Borrows from the underlying Vulkan call data — do not retain references
/// outside the callback. Copy any string fields you want to keep.
pub struct DebugMessage<'a> {
    pub severity: DebugMessageSeverity,
    pub message_type: DebugMessageType,
    /// The free-form message text from the implementation. Always present.
    pub message: &'a str,
    /// Optional message ID name (e.g. `"VUID-vkCreateBuffer-size-00912"`).
    pub message_id_name: Option<&'a str>,
    /// Optional integer message ID.
    pub message_id_number: i32,
}

/// User callback type for debug messages.
///
/// Must be `Send + Sync` because Vulkan may invoke it from any thread.
pub type DebugCallback = dyn Fn(&DebugMessage<'_>) + Send + Sync + 'static;

/// A pre-built default callback that prints WARNING and ERROR messages to
/// `stderr` via `eprintln!`. Suitable for development.
pub fn default_callback() -> Box<DebugCallback> {
    Box::new(|msg: &DebugMessage<'_>| {
        if msg.severity.contains(DebugMessageSeverity::WARNING)
            || msg.severity.contains(DebugMessageSeverity::ERROR)
        {
            eprintln!(
                "[VK {}/{}] {}",
                msg.severity.label(),
                msg.message_type.label(),
                msg.message
            );
        }
    })
}

/// The proper Vulkan signature for the debug-utils callback. The
/// auto-generated `PFN_vkDebugUtilsMessengerCallbackEXT` typedef is empty
/// because vk.xml describes function pointer signatures inline rather than
/// via the same parser path as commands; we declare the correct signature
/// here and transmute on the way out.
pub(crate) type RealDebugCallbackFn = unsafe extern "system" fn(
    severity: u32,
    message_types: u32,
    p_callback_data: *const VkDebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut std::ffi::c_void,
) -> u32;

/// The trampoline that converts the C callback into a Rust closure call.
///
/// `p_user_data` is a `*mut Box<DebugCallback>` we placed there at messenger
/// creation time. We must not free it here — it's owned by the
/// [`DebugMessenger`] guard.
pub(crate) unsafe extern "system" fn trampoline(
    severity: u32,
    message_types: u32,
    p_callback_data: *const VkDebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut std::ffi::c_void,
) -> u32 {
    if p_callback_data.is_null() || p_user_data.is_null() {
        return 0;
    }
    // Safety: we created p_user_data as a `Box<Box<DebugCallback>>` leaked
    // pointer; the caller (this trampoline) is invoked while the messenger
    // is alive and the box has not been freed. We deref one level of the
    // outer box to obtain the inner `&DebugCallback` directly so we don't
    // have to materialise a `&Box<dyn ...>` (which clippy dislikes).
    let cb: &DebugCallback =
        unsafe { (*(p_user_data as *const Box<DebugCallback>)).as_ref() };

    // Safety: Vulkan promises this pointer is valid for the duration of
    // the callback.
    let data = unsafe { &*p_callback_data };

    let message = if data.pMessage.is_null() {
        ""
    } else {
        // Safety: pMessage is a NUL-terminated UTF-8 string per spec.
        unsafe { CStr::from_ptr(data.pMessage) }
            .to_str()
            .unwrap_or("<invalid utf-8>")
    };

    let message_id_name = if data.pMessageIdName.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(data.pMessageIdName) }.to_str().ok()
    };

    let msg = DebugMessage {
        severity: DebugMessageSeverity(severity),
        message_type: DebugMessageType(message_types),
        message,
        message_id_name,
        message_id_number: data.messageIdNumber,
    };

    cb(&msg);
    0 // VK_FALSE — never abort the call
}
