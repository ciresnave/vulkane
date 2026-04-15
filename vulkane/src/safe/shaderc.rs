//! Optional GLSL/HLSL -> SPIR-V compilation via the [`shaderc`] crate
//! (Khronos reference glslang).
//!
//! This module is only compiled when the `shaderc` Cargo feature is
//! enabled:
//!
//! ```toml
//! [dependencies]
//! vulkane = { version = "0.4", features = ["shaderc"] }
//! ```
//!
//! # When to use this vs. `naga`
//!
//! The [`naga`](crate::safe::naga) feature is pure Rust and covers a
//! large subset of modern GLSL. `shaderc` wraps Khronos's reference
//! glslang compiler and is the gold standard for GLSL -> SPIR-V —
//! pick it when you need full GLSL support (including `#include`,
//! `GL_*` extensions, legacy shaders, or HLSL via
//! [`shaderc::SourceLanguage::HLSL`]).
//!
//! # Build requirements
//!
//! `shaderc-rs` tries these in order:
//!
//! 1. `SHADERC_LIB_DIR` env var
//! 2. `VULKAN_SDK` env var (set by installing the LunarG Vulkan SDK)
//! 3. `pkg-config` / system libraries
//! 4. Fallback: builds glslang from C++ source (requires CMake, Python,
//!    and a working C++ toolchain)
//!
//! The easiest path is to install the Vulkan SDK; `shaderc-rs` will
//! find and link the prebuilt `libshaderc_combined` shipped with it.

use super::Error;
pub use shaderc::{ShaderKind, SourceLanguage, TargetEnv};

/// Errors that can occur during shaderc compilation.
#[derive(Debug)]
pub enum ShadercError {
    /// The `shaderc::Compiler` could not be constructed. This usually
    /// means libshaderc is present but failed to initialize.
    CompilerInit,
    /// Creating a `CompileOptions` object failed.
    OptionsInit,
    /// Compilation failed — the string contains glslang's diagnostic
    /// output (file name, line, column, and message for each error).
    Compile(String),
}

impl std::fmt::Display for ShadercError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompilerInit => write!(f, "failed to initialize shaderc compiler"),
            Self::OptionsInit => write!(f, "failed to initialize shaderc compile options"),
            Self::Compile(s) => write!(f, "shaderc compilation failed:\n{s}"),
        }
    }
}

impl std::error::Error for ShadercError {}

impl From<ShadercError> for Error {
    fn from(e: ShadercError) -> Self {
        Error::ShadercCompile(e.to_string())
    }
}

/// Compile a GLSL (or HLSL) source string to a SPIR-V word vector.
///
/// * `source`      — the shader source text.
/// * `kind`        — the shader stage ([`ShaderKind::Vertex`],
///   [`ShaderKind::Fragment`], [`ShaderKind::Compute`], etc.).
/// * `file_name`   — a virtual filename used in diagnostic messages
///   and as the base for `#include` resolution. Does not need to
///   exist on disk.
/// * `entry_point` — the entry-point function name (commonly `"main"`
///   for GLSL; varies for HLSL).
///
/// The returned `Vec<u32>` can be passed directly to
/// [`crate::safe::ShaderModule::from_spirv`].
///
/// # Example
///
/// ```ignore
/// use vulkane::safe::shaderc::{compile_glsl, ShaderKind};
///
/// let glsl = r#"
///     #version 450
///     layout(local_size_x = 64) in;
///     layout(set = 0, binding = 0, std430) buffer Data { uint values[]; };
///     void main() { values[gl_GlobalInvocationID.x] *= 2; }
/// "#;
///
/// let spirv = compile_glsl(glsl, ShaderKind::Compute, "doubler.comp", "main")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compile_glsl(
    source: &str,
    kind: ShaderKind,
    file_name: &str,
    entry_point: &str,
) -> Result<Vec<u32>, ShadercError> {
    let compiler = shaderc::Compiler::new().ok_or(ShadercError::CompilerInit)?;
    let options = shaderc::CompileOptions::new().ok_or(ShadercError::OptionsInit)?;

    let artifact = compiler
        .compile_into_spirv(source, kind, file_name, entry_point, Some(&options))
        .map_err(|e| ShadercError::Compile(e.to_string()))?;

    Ok(artifact.as_binary().to_vec())
}

/// Compile a source string to SPIR-V with full control over compiler
/// options (optimization level, target environment, macro defines,
/// HLSL vs. GLSL, etc.).
///
/// Use [`compile_glsl`] for the common case; reach for this when you
/// need non-default options.
///
/// The `configure` closure runs before compilation and receives a
/// mutable [`shaderc::CompileOptions`] builder.
///
/// # Example — HLSL input, size-optimized
///
/// ```ignore
/// use vulkane::safe::shaderc::{compile_with_options, ShaderKind, SourceLanguage};
/// use shaderc::OptimizationLevel;
///
/// let spirv = compile_with_options(
///     hlsl_source,
///     ShaderKind::Fragment,
///     "shader.hlsl",
///     "main",
///     |opts| {
///         opts.set_source_language(SourceLanguage::HLSL);
///         opts.set_optimization_level(OptimizationLevel::Size);
///     },
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compile_with_options<F>(
    source: &str,
    kind: ShaderKind,
    file_name: &str,
    entry_point: &str,
    configure: F,
) -> Result<Vec<u32>, ShadercError>
where
    F: FnOnce(&mut shaderc::CompileOptions<'_>),
{
    let compiler = shaderc::Compiler::new().ok_or(ShadercError::CompilerInit)?;
    let mut options = shaderc::CompileOptions::new().ok_or(ShadercError::OptionsInit)?;
    configure(&mut options);

    let artifact = compiler
        .compile_into_spirv(source, kind, file_name, entry_point, Some(&options))
        .map_err(|e| ShadercError::Compile(e.to_string()))?;

    Ok(artifact.as_binary().to_vec())
}
