//! Registry for precompiled SPIR-V shader modules.
//!
//! Applications that ship their shaders as precompiled `.spv` artifacts
//! (embedded via [`include_bytes!`] or shipped alongside the binary)
//! benefit from a small, shared abstraction for looking up and loading
//! them by name — with an optional runtime disk-override for shader
//! developers iterating without a full rebuild.
//!
//! # Example
//!
//! ```no_run
//! use vulkane::safe::{ShaderRegistry, ShaderSource};
//!
//! // Embedded at compile time. `include_bytes!` resolves paths relative
//! // to the file that invokes it, so the macro call lives in your crate.
//! const EMBEDDED: &[ShaderSource] = &[
//!     ShaderSource { name: "doubler", spv: include_bytes!("shaders/doubler.spv") },
//!     ShaderSource { name: "reduce",  spv: include_bytes!("shaders/reduce.spv")  },
//! ];
//!
//! let registry = ShaderRegistry::new()
//!     .with_embedded(EMBEDDED)
//!     .with_env_override("MY_APP_SHADER_OVERRIDE_DIR");
//!
//! // Later, when you have a Device:
//! # fn use_registry(
//! #     registry: &ShaderRegistry,
//! #     device: &vulkane::safe::Device,
//! # ) -> Result<(), Box<dyn std::error::Error>> {
//! let module = registry.load_module(device, "doubler")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Lookup order
//!
//! `load(name)` checks sources in this order:
//!
//! 1. **Environment override** (if configured via
//!    [`ShaderRegistry::with_env_override`]) — looks for
//!    `$VAR/<name>.spv` on disk.
//! 2. **Embedded table** (configured via
//!    [`ShaderRegistry::with_embedded`]).
//!
//! If the override directory exists but the file is missing, the
//! registry falls through to the embedded table — so you only need to
//! place overrides for the specific shaders you're iterating on.

use super::{Device, Error, Result, ShaderModule};
use std::borrow::Cow;
use std::path::PathBuf;

/// A precompiled SPIR-V shader identified by name.
///
/// The typical way to populate a `ShaderSource` is via
/// [`include_bytes!`], which produces a `&'static [u8]`:
///
/// ```ignore
/// ShaderSource { name: "my_shader", spv: include_bytes!("my_shader.spv") }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ShaderSource {
    /// Logical shader name used as the lookup key.
    pub name: &'static str,
    /// Raw SPIR-V bytes (little-endian 32-bit words on the wire).
    pub spv: &'static [u8],
}

/// Errors returned by [`ShaderRegistry`] lookup and load methods.
#[derive(Debug)]
pub enum ShaderLoadError {
    /// No shader with the given name exists in either the override
    /// directory or the embedded table.
    NotFound(String),
    /// Reading an override file from disk failed.
    Io {
        /// Shader name that was being loaded.
        name: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// SPIR-V byte length is not a multiple of 4, so it cannot be a
    /// valid SPIR-V binary.
    MalformedSpirv {
        /// Shader name that was being loaded.
        name: String,
        /// Actual byte length received.
        byte_len: usize,
    },
}

impl std::fmt::Display for ShaderLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "shader not found: {name}"),
            Self::Io { name, source } => {
                write!(f, "failed to read shader {name} from override directory: {source}")
            }
            Self::MalformedSpirv { name, byte_len } => write!(
                f,
                "shader {name} has malformed SPIR-V: {byte_len} bytes is not a multiple of 4",
            ),
        }
    }
}

impl std::error::Error for ShaderLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<ShaderLoadError> for Error {
    fn from(e: ShaderLoadError) -> Self {
        Error::ShaderLoad(e)
    }
}

/// Registry mapping shader names to SPIR-V bytes.
///
/// Build one at application startup with the list of embedded shaders
/// and (optionally) an environment variable for disk overrides, then
/// pass it to the code paths that need shader modules.
#[derive(Default)]
pub struct ShaderRegistry {
    embedded: &'static [ShaderSource],
    env_override: Option<&'static str>,
}

impl ShaderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedded shader table. The slice must be `'static` —
    /// typically a `const &[ShaderSource]` in the consumer crate built
    /// from [`include_bytes!`] calls.
    pub fn with_embedded(mut self, shaders: &'static [ShaderSource]) -> Self {
        self.embedded = shaders;
        self
    }

    /// Enable runtime disk overrides via the named environment
    /// variable. When set to a directory path, the registry will look
    /// for `<dir>/<shader_name>.spv` **before** consulting the
    /// embedded table.
    ///
    /// If the env var is unset, points to a nonexistent path, or the
    /// specific override file is missing, the registry falls through
    /// to the embedded table — so you only need to place overrides for
    /// the shaders you're iterating on.
    pub fn with_env_override(mut self, var: &'static str) -> Self {
        self.env_override = Some(var);
        self
    }

    /// Resolve a shader by name and return its SPIR-V bytes.
    ///
    /// Borrowed from the embedded table if possible, or owned if read
    /// from the override directory.
    pub fn load(&self, name: &str) -> std::result::Result<Cow<'_, [u8]>, ShaderLoadError> {
        if let Some(var) = self.env_override {
            if let Some(dir) = override_dir(var) {
                let path = dir.join(format!("{name}.spv"));
                if path.exists() {
                    return std::fs::read(&path).map(Cow::Owned).map_err(|source| {
                        ShaderLoadError::Io {
                            name: name.to_owned(),
                            source,
                        }
                    });
                }
            }
        }
        self.embedded
            .iter()
            .find(|s| s.name == name)
            .map(|s| Cow::Borrowed(s.spv))
            .ok_or_else(|| ShaderLoadError::NotFound(name.to_owned()))
    }

    /// Resolve a shader by name and return its SPIR-V words, ready to
    /// pass to [`ShaderModule::from_spirv`].
    pub fn load_words(&self, name: &str) -> std::result::Result<Vec<u32>, ShaderLoadError> {
        let bytes = self.load(name)?;
        if bytes.len() % 4 != 0 {
            return Err(ShaderLoadError::MalformedSpirv {
                name: name.to_owned(),
                byte_len: bytes.len(),
            });
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Resolve a shader by name and create a live [`ShaderModule`] on
    /// the given device.
    ///
    /// Convenience over `registry.load(name).and_then(|bytes|
    /// ShaderModule::from_spirv_bytes(device, &bytes))`.
    pub fn load_module(&self, device: &Device, name: &str) -> Result<ShaderModule> {
        let bytes = self.load(name)?;
        ShaderModule::from_spirv_bytes(device, &bytes)
    }
}

fn override_dir(var: &str) -> Option<PathBuf> {
    let raw = std::env::var_os(var)?;
    let path = PathBuf::from(raw);
    path.metadata().ok().filter(|m| m.is_dir()).map(|_| path)
}
