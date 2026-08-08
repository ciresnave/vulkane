//! Optional GLSL -> SPIR-V compilation via the [`naga`] crate.
//!
//! This module is only compiled when the `naga` Cargo feature is enabled:
//!
//! ```toml
//! [dependencies]
//! vulkane = { version = "0.1", features = ["naga"] }
//! ```
//!
//! It provides one convenience function, [`compile_glsl`], that takes a
//! GLSL source string and a shader stage and returns a `Vec<u32>` of SPIR-V
//! words ready to pass to [`crate::safe::ShaderModule::from_spirv`].
//!
//! Most production projects pre-compile shaders with `glslc` (or another
//! offline tool) and ship the resulting `.spv` files. Use this module if
//! you want runtime compilation, hot-reloading, or to embed GLSL source in
//! your binary.

use super::Error;
use naga::back::spv;
use naga::front::{glsl, wgsl};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{Module, ShaderStage};

/// Errors that can occur during GLSL compilation.
#[derive(Debug)]
pub enum NagaError {
    /// One or more GLSL parse errors.
    Parse(String),
    /// Naga's IR validator rejected the parsed module.
    Validation(String),
    /// SPIR-V emission failed.
    SpvOut(String),
}

impl std::fmt::Display for NagaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(s) => write!(f, "GLSL parse error: {s}"),
            Self::Validation(s) => write!(f, "Naga validation error: {s}"),
            Self::SpvOut(s) => write!(f, "SPIR-V emission error: {s}"),
        }
    }
}

impl std::error::Error for NagaError {}

impl From<NagaError> for Error {
    fn from(e: NagaError) -> Self {
        // `NagaCompile` carries the rendered message rather than the typed
        // `NagaError`, matching the shape of the sibling `ShadercCompile` and
        // `SlangCompile` variants so all three shader front-ends surface the
        // same way. `NagaError`'s `Display` names the failing stage, so the
        // parse/validation/emission distinction survives the conversion.
        Error::NagaCompile(e.to_string())
    }
}

/// Compile a GLSL source string to a SPIR-V word vector.
///
/// `stage` selects the shader stage (vertex, fragment, compute, etc.).
/// The returned `Vec<u32>` can be passed directly to
/// [`crate::safe::ShaderModule::from_spirv`].
///
/// # Example
///
/// ```ignore
/// use vulkane::safe::naga::compile_glsl;
/// use naga::ShaderStage;
///
/// let glsl = r#"
///     #version 450
///     layout(local_size_x = 64) in;
///     layout(set = 0, binding = 0, std430) buffer Data { uint values[]; };
///     void main() { values[gl_GlobalInvocationID.x] *= 2; }
/// "#;
///
/// let spirv = compile_glsl(glsl, ShaderStage::Compute)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compile_glsl(source: &str, stage: ShaderStage) -> Result<Vec<u32>, NagaError> {
    // Parse GLSL -> Naga IR
    let mut parser = glsl::Frontend::default();
    let module: Module = parser
        .parse(&glsl::Options::from(stage), source)
        .map_err(|errors| NagaError::Parse(format!("{errors:?}")))?;

    spirv_from_module(module)
}

/// Compile a WGSL source string to a SPIR-V word vector.
///
/// WGSL (WebGPU Shading Language) is the canonical input format for
/// [`naga`]. Unlike the GLSL frontend, the WGSL frontend handles
/// combined image samplers (`texture_2d<f32>` + `sampler`) cleanly,
/// which makes it the right choice for textured rendering examples.
///
/// All entry points in the WGSL module are translated to a single
/// multi-entry-point SPIR-V module. The caller selects which entry
/// point each `VkPipelineShaderStageCreateInfo` runs by passing the
/// matching `pName` (e.g. `"vs_main"` for the vertex stage and
/// `"fs_main"` for the fragment stage).
///
/// # Example
///
/// ```ignore
/// use vulkane::safe::naga::compile_wgsl;
///
/// let wgsl = r#"
///     @group(0) @binding(0) var tex: texture_2d<f32>;
///     @group(0) @binding(1) var samp: sampler;
///
///     @fragment
///     fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
///         return textureSample(tex, samp, uv);
///     }
/// "#;
///
/// let spirv = compile_wgsl(wgsl)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compile_wgsl(source: &str) -> Result<Vec<u32>, NagaError> {
    let module = wgsl::parse_str(source).map_err(|e| NagaError::Parse(e.emit_to_string(source)))?;
    spirv_from_module(module)
}

fn spirv_from_module(module: Module) -> Result<Vec<u32>, NagaError> {
    // Validate the IR
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| NagaError::Validation(format!("{e:?}")))?;

    // Emit SPIR-V. Target Vulkan 1.0 for maximum compatibility unless the
    // caller wants more.
    let spv_options = spv::Options {
        lang_version: (1, 0),
        ..spv::Options::default()
    };

    let words = spv::write_vec(&module, &info, &spv_options, None)
        .map_err(|e| NagaError::SpvOut(format!("{e:?}")))?;

    Ok(words)
}
