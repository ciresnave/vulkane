//! Ergonomic conversion of `VkResult` into Rust's `Result` type.
//!
//! Vulkan functions return a `VkResult` enum where:
//!
//! - `SUCCESS` (0) means the call completed.
//! - **Positive** values are non-fatal status codes (`NOT_READY`, `TIMEOUT`,
//!   `INCOMPLETE`, `SUBOPTIMAL_KHR`, etc.) — the call did something useful but
//!   not necessarily what you asked for.
//! - **Negative** values are errors (`ERROR_OUT_OF_HOST_MEMORY`, `ERROR_DEVICE_LOST`,
//!   etc.) — the call failed and any output parameters are invalid.
//!
//! Spock provides:
//!
//! - The [`VkResultExt`] extension trait, with `into_result()`, `is_success()`,
//!   and `is_error()` methods.
//! - A `std::error::Error` implementation for [`VkResult`] so you can use `?`
//!   in functions returning `Result<_, Box<dyn Error>>`.
//! - The [`vk_check!`] macro for one-line Vulkan call validation.
//!
//! `into_result()` only treats `SUCCESS` as `Ok`. If you need to distinguish
//! between non-fatal status codes (like `INCOMPLETE`) and outright errors,
//! use the raw `VkResult` value directly.
//!
//! # Example: propagating errors with `?`
//!
//! ```ignore
//! use spock::raw::bindings::*;
//! use spock::raw::VkResultExt;
//!
//! fn create_instance(entry: &VkEntryDispatchTable) -> Result<VkInstance, Box<dyn std::error::Error>> {
//!     let info = VkInstanceCreateInfo::default();
//!     let mut instance: VkInstance = std::ptr::null_mut();
//!     let create = entry.vkCreateInstance.ok_or("vkCreateInstance not loaded")?;
//!     unsafe { create(&info, std::ptr::null(), &mut instance) }.into_result()?;
//!     Ok(instance)
//! }
//! ```

use crate::raw::bindings::VkResult;

/// Extension trait that converts `VkResult` into a Rust `Result`.
pub trait VkResultExt {
    /// Convert `VkResult::SUCCESS` to `Ok(())` and any other value to `Err(self)`.
    fn into_result(self) -> Result<(), VkResult>;

    /// Returns true if the result indicates success (SUCCESS).
    /// Note that some "non-error" status codes like NOT_READY, TIMEOUT, INCOMPLETE,
    /// and SUBOPTIMAL_KHR are technically positive but indicate work was not completed.
    fn is_success(self) -> bool;

    /// Returns true if the result indicates an error (any negative value).
    fn is_error(self) -> bool;
}

impl VkResultExt for VkResult {
    #[inline]
    fn into_result(self) -> Result<(), VkResult> {
        if self == VkResult::SUCCESS {
            Ok(())
        } else {
            Err(self)
        }
    }

    #[inline]
    fn is_success(self) -> bool {
        self == VkResult::SUCCESS
    }

    #[inline]
    fn is_error(self) -> bool {
        (self as i32) < 0
    }
}

// Implement std::error::Error for VkResult so it works with `?` in functions
// returning Box<dyn std::error::Error>. The generated VkResult already has
// Debug + Display impls from the codegen.
impl std::error::Error for VkResult {}

/// Convenience macro that calls a Vulkan function and propagates errors via `?`.
///
/// # Example
///
/// ```ignore
/// use spock::vk_check;
/// // Inside an unsafe fn that returns Result<_, VkResult>:
/// vk_check!(create_instance(&info, std::ptr::null(), &mut instance))?;
/// ```
#[macro_export]
macro_rules! vk_check {
    ($call:expr) => {{
        use $crate::raw::result::VkResultExt;
        ($call).into_result()
    }};
}
