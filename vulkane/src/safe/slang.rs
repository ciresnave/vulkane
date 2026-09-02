//! Optional Slang -> SPIR-V compilation via the [`shader-slang`] crate.
//!
//! This module is only compiled when the `slang` Cargo feature is enabled:
//!
//! ```toml
//! [dependencies]
//! vulkane = { version = "0.4", features = ["slang"] }
//! ```
//!
//! # Why Slang?
//!
//! Slang is Khronos's modern shading language. Compared with GLSL/HLSL it
//! adds:
//!
//! - **Modules, generics, and interfaces** — real code reuse in shaders.
//! - **Built-in automatic differentiation** — mark a function
//!   `[Differentiable]` and the compiler generates forward (`__fwd_diff`)
//!   and backward (`__bwd_diff`) variants. A single kernel definition
//!   yields both the forward and backward SPIR-V for ML workloads.
//! - **Explicit entry points** with attribute-driven stage selection.
//!
//! # API shape
//!
//! Slang compilation is stateful. Unlike `naga` and `shaderc`, which
//! take a source string and hand back SPIR-V in one call, Slang wants
//! you to:
//!
//! 1. Create a **session** ([`SlangSession::new`] /
//!    [`SlangSession::with_search_paths`]) — configures SPIR-V target
//!    and the directories Slang searches for `.slang` files.
//! 2. **Load a module** from disk ([`SlangSession::load_file`]) —
//!    parses and compiles a `.slang` source unit, resolving any
//!    `import` statements through the search paths.
//! 3. **Request entry-point code**
//!    ([`SlangModule::compile_entry_point`]) — returns a `Vec<u32>`
//!    of SPIR-V words per entry point.
//!
//! For the common one-shot case, [`compile_slang_file`] collapses all
//! three steps into a single call.
//!
//! # Autodiff example
//!
//! Given `kernels/autodiff.slang`:
//!
//! ```text
//! [Differentiable]
//! float square(float x) { return x * x; }
//!
//! [shader("compute")]
//! [numthreads(1, 1, 1)]
//! void forward(uint3 id : SV_DispatchThreadID) { /* ... */ }
//!
//! [shader("compute")]
//! [numthreads(1, 1, 1)]
//! void backward(uint3 id : SV_DispatchThreadID) { /* uses __bwd_diff(square) */ }
//! ```
//!
//! ```ignore
//! use vulkane::safe::slang::SlangSession;
//!
//! let session = SlangSession::with_search_paths(&["kernels"])?;
//! let module  = session.load_file("autodiff")?;     // finds kernels/autodiff.slang
//! let fwd     = module.compile_entry_point("forward")?;   // Vec<u32> SPIR-V
//! let bwd     = module.compile_entry_point("backward")?;  // Vec<u32> SPIR-V
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Both `fwd` and `bwd` can be passed straight to
//! [`crate::safe::ShaderModule::from_spirv`]. Parsing happens once;
//! each `compile_entry_point` call just emits bytecode for one entry.
//!
//! # Note on inline source
//!
//! `shader-slang` 0.1.0 on crates.io does not expose a source-string
//! loader (only `load_module` from disk). When a newer version ships
//! that re-exposes `loadModuleFromSourceString`, this module will gain
//! a `SlangSession::load_source` method.
//!
//! # Build requirements
//!
//! `shader-slang` locates the Slang compiler via:
//!
//! 1. **`VULKAN_SDK`** env var — installing the
//!    [LunarG Vulkan SDK](https://vulkan.lunarg.com/) ships `slangc`
//!    and `slang.dll`/`libslang.so`. **Easiest path.**
//! 2. **`SLANG_DIR`** env var — a standalone Slang install from
//!    <https://github.com/shader-slang/slang/releases>.
//! 3. **`SLANG_INCLUDE_DIR`** + **`SLANG_LIB_DIR`** — split paths.
//!
//! At runtime, `slang.dll` / `libslang.so` / `libslang.dylib` must be
//! discoverable (same directory as the executable, or on the library
//! search path).

use super::Error;
use slang::Downcast;
use std::ffi::CString;

pub use slang::{CompileTarget, OptimizationLevel, Stage};

/// Errors that can occur during Slang compilation.
#[derive(Debug)]
pub enum SlangError {
    /// Couldn't initialize the Slang global compiler session. Usually
    /// means the Slang runtime library could not be loaded.
    GlobalInit,
    /// The compiler session could not be created from the given
    /// targets / search paths.
    SessionCreate(String),
    /// Parsing / compiling the source module failed. The string
    /// contains Slang's diagnostic output.
    LoadModule(String),
    /// No entry point with the given name exists in the module.
    EntryPointNotFound(String),
    /// Composing module + entry point into a program failed.
    Composite(String),
    /// Linking the composite component failed.
    Link(String),
    /// Retrieving the compiled entry-point SPIR-V failed.
    EntryPointCode(String),
    /// The returned SPIR-V blob is not a multiple of 4 bytes.
    MalformedSpirv(String),
}

impl std::fmt::Display for SlangError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GlobalInit => write!(f, "failed to initialize Slang global session"),
            Self::SessionCreate(s) => write!(f, "Slang session creation failed: {s}"),
            Self::LoadModule(s) => write!(f, "Slang module compilation failed:\n{s}"),
            Self::EntryPointNotFound(s) => write!(f, "Slang entry point not found: {s}"),
            Self::Composite(s) => write!(f, "Slang component composition failed: {s}"),
            Self::Link(s) => write!(f, "Slang linking failed: {s}"),
            Self::EntryPointCode(s) => write!(f, "Slang entry-point code retrieval failed: {s}"),
            Self::MalformedSpirv(s) => write!(f, "Slang produced malformed SPIR-V: {s}"),
        }
    }
}

impl std::error::Error for SlangError {}

impl From<SlangError> for Error {
    fn from(e: SlangError) -> Self {
        Error::SlangCompile(e.to_string())
    }
}

/// A Slang compiler session — holds a [`slang::GlobalSession`] and the
/// configured [`slang::Session`] targeting SPIR-V for Vulkan.
///
/// Create one session per shader library. Modules loaded within a
/// session share search paths and compiler options.
pub struct SlangSession {
    _global: slang::GlobalSession,
    session: slang::Session,
    // Owned CStrings backing the search-path pointers handed to Slang.
    // Kept alive for the whole session in case Slang only retained the
    // pointers rather than copying the strings.
    _search_paths: Vec<CString>,
}

impl SlangSession {
    /// Create a session with no search paths.
    ///
    /// [`load_file`](Self::load_file) will only resolve Slang modules
    /// by absolute path with no search-path lookup.
    pub fn new() -> Result<Self, SlangError> {
        Self::with_search_paths(&[])
    }

    /// Create a session with the given include / import search paths.
    ///
    /// Paths are used when loading a module by name (either via
    /// [`load_file`](Self::load_file) or when one loaded module does
    /// `import Foo;` inside Slang source) — Slang looks for `Foo.slang`
    /// in each path in order.
    pub fn with_search_paths(paths: &[&str]) -> Result<Self, SlangError> {
        let global = slang::GlobalSession::new().ok_or(SlangError::GlobalInit)?;

        let search_paths: Vec<CString> = paths
            .iter()
            .map(|p| CString::new(*p).expect("search path contains NUL byte"))
            .collect();
        let search_path_ptrs: Vec<*const i8> = search_paths.iter().map(|s| s.as_ptr()).collect();

        let target_desc = slang::TargetDesc::default()
            .format(slang::CompileTarget::Spirv)
            .profile(global.find_profile("glsl_450"));
        let targets = [target_desc];

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_path_ptrs);

        let session = global
            .create_session(&session_desc)
            .ok_or_else(|| SlangError::SessionCreate("create_session returned None".into()))?;

        Ok(Self {
            _global: global,
            session,
            _search_paths: search_paths,
        })
    }

    /// Load a Slang module by name, resolving through the session's
    /// search paths.
    ///
    /// The `name` typically omits the `.slang` extension — Slang adds
    /// it when searching. Pass e.g. `"kernels/autodiff"` to load
    /// `kernels/autodiff.slang` from the first matching search path.
    pub fn load_file(&self, name: &str) -> Result<SlangModule<'_>, SlangError> {
        let module = self
            .session
            .load_module(name)
            .map_err(|e| SlangError::LoadModule(e.to_string()))?;
        Ok(SlangModule {
            module,
            session: &self.session,
        })
    }
}

/// A compiled Slang module, bound to its originating [`SlangSession`].
///
/// One module can yield many entry-point SPIR-V blobs — call
/// [`compile_entry_point`](Self::compile_entry_point) once per entry
/// point you need. For autodiff, define both the forward and backward
/// kernels as `[shader("compute")]` entry points in the module and
/// request each by name.
pub struct SlangModule<'sess> {
    module: slang::Module,
    session: &'sess slang::Session,
}

impl<'sess> SlangModule<'sess> {
    /// Compile the named entry point and return its SPIR-V words,
    /// ready to pass to [`crate::safe::ShaderModule::from_spirv`].
    pub fn compile_entry_point(&self, name: &str) -> Result<Vec<u32>, SlangError> {
        let entry_point = self
            .module
            .find_entry_point_by_name(name)
            .ok_or_else(|| SlangError::EntryPointNotFound(name.to_owned()))?;

        // Compose module + entry point into a program, then link it.
        // Module / EntryPoint are COM-refcounted; `.downcast().clone()`
        // upcasts to `ComponentType` without consuming the originals.
        let program = self
            .session
            .create_composite_component_type(&[
                self.module.downcast().clone(),
                entry_point.downcast().clone(),
            ])
            .map_err(|e| SlangError::Composite(e.to_string()))?;

        let linked = program
            .link()
            .map_err(|e| SlangError::Link(e.to_string()))?;

        let blob = linked
            .entry_point_code(0, 0)
            .map_err(|e| SlangError::EntryPointCode(e.to_string()))?;

        // SPIR-V on the wire is little-endian 32-bit words.
        let bytes = blob.as_slice();
        if bytes.len() % 4 != 0 {
            return Err(SlangError::MalformedSpirv(format!(
                "blob length {} is not a multiple of 4",
                bytes.len()
            )));
        }
        // `as_chunks` rather than `chunks_exact(4)`: clippy's
        // `chunks_exact_to_as_chunks` fires on a constant chunk size, and it is
        // MSRV-gated at 1.88 -- so it could only ever be seen by a lint run at
        // this crate's floor, and this file had never met one. It was caught the
        // first time the `slang` feature was linted at all.
        //
        // `as_chunks::<4>` yields `&[[u8; 4]]`, which drops the `c[0]..c[3]`
        // indexing and the bounds checks with it. The length was validated as a
        // multiple of 4 above, so the remainder is empty by construction.
        let (words, remainder) = bytes.as_chunks::<4>();
        debug_assert!(
            remainder.is_empty(),
            "length was checked as a multiple of 4 above"
        );
        Ok(words.iter().copied().map(u32::from_le_bytes).collect())
    }
}

/// One-shot convenience: create a session rooted at `search_dir`, load
/// `module_name`, compile `entry_point`, return SPIR-V words.
///
/// For workflows that compile multiple entry points from a single
/// source (e.g. forward + backward autodiff kernels), use
/// [`SlangSession`] directly so the module is only parsed once.
///
/// # Example
///
/// ```ignore
/// use vulkane::safe::slang::compile_slang_file;
///
/// // Loads ./kernels/trivial.slang and compiles its `main` entry point.
/// let spirv = compile_slang_file("kernels", "trivial", "main")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compile_slang_file(
    search_dir: &str,
    module_name: &str,
    entry_point: &str,
) -> Result<Vec<u32>, SlangError> {
    let session = SlangSession::with_search_paths(&[search_dir])?;
    let module = session.load_file(module_name)?;
    module.compile_entry_point(entry_point)
}
