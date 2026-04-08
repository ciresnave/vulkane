//! Safe wrapper for `VkShaderModule`.

use super::device::DeviceInner;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// A safe wrapper around `VkShaderModule`.
///
/// A shader module is a chunk of compiled SPIR-V bytecode loaded into the
/// driver. To produce SPIR-V, you can:
///
/// - Pre-compile GLSL with `glslc -O shader.comp -o shader.spv` and load
///   the resulting bytes into a `&[u32]`.
/// - Use the optional `naga` Cargo feature to compile GLSL at runtime via
///   [`crate::safe::naga::compile_glsl`].
/// - Use any other SPIR-V producer (rust-gpu, slang, etc.) — spock takes
///   any valid SPIR-V word slice.
///
/// Shader modules are destroyed automatically on drop.
pub struct ShaderModule {
    pub(crate) handle: VkShaderModule,
    pub(crate) device: Arc<DeviceInner>,
}

impl ShaderModule {
    /// Create a shader module from a slice of SPIR-V words.
    ///
    /// `code` must be a valid SPIR-V binary (the bytes start with the SPIR-V
    /// magic number `0x07230203`). Length is in `u32` words, not bytes — most
    /// SPIR-V files are stored as little-endian byte streams that you should
    /// read via `bytemuck::cast_slice` or similar before passing here.
    pub fn from_spirv(device: &Device, code: &[u32]) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateShaderModule
            .ok_or(Error::MissingFunction("vkCreateShaderModule"))?;

        let info = VkShaderModuleCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            // VkShaderModuleCreateInfo::codeSize is in BYTES, not words.
            codeSize: std::mem::size_of_val(code),
            pCode: code.as_ptr(),
            ..Default::default()
        };

        let mut handle: VkShaderModule = 0;
        // Safety: info is valid for the call, code lives for the call,
        // device handle is valid.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Convenience: create a shader module from a slice of bytes containing
    /// little-endian SPIR-V. The byte length must be a multiple of 4.
    pub fn from_spirv_bytes(device: &Device, bytes: &[u8]) -> Result<Self> {
        if bytes.len() % 4 != 0 {
            return Err(Error::Vk(VkResult::ERROR_INITIALIZATION_FAILED));
        }
        // Safety: alignment of u32 is 4 and we just checked length is a
        // multiple of 4. We also do an aligned copy to be safe regardless of
        // input alignment.
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::from_spirv(device, &words)
    }

    /// Returns the raw `VkShaderModule` handle.
    pub fn raw(&self) -> VkShaderModule {
        self.handle
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyShaderModule {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}
