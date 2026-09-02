//! Build script for Vulkane - generates Vulkan bindings from vk.xml specification
//!
//! vk.xml resolution order — FOUR routes, and this list used to name three.
//! The one it omitted is the one that serves everyone who depends on this crate,
//! which made the documented order the inverse of the useful one: it featured a
//! path only this workspace can satisfy and left out the bundled copy that makes
//! a plain `cargo add vulkane` work offline.
//!
//! 1. `VK_XML_PATH` environment variable. Use an ABSOLUTE path: a relative one
//!    is tried as given (resolved against this crate's directory) and then
//!    against its parent, which for a dependency is cargo's registry cache --
//!    so neither base is one a dependent controls. It reads as "workspace
//!    root" only from inside this repository.
//! 2. Bundled copy at `<CARGO_MANIFEST_DIR>/vk.xml`. **Ships inside the
//!    published crate**, so dependents, docs.rs and other sandboxed builds
//!    resolve without network access or configuration. This is the route the
//!    overwhelming majority of builds take.
//! 3. Workspace copy at `../spec/registry/Vulkan-Docs/xml/vk.xml`, relative to
//!    this crate. Reachable only from a checkout of this repository — for a
//!    dependency, `..` is cargo's registry cache.
//! 4. Auto-download from Khronos GitHub (requires the `fetch-spec` feature)
//!    - Set `VK_VERSION` to download a specific version (e.g. `VK_VERSION=1.3.250`)
//!    - Without `VK_VERSION`, downloads the latest from the main branch
//!
//! Because route 2 is present in every published version, `fetch-spec` is a
//! fallback rather than a requirement.
//!
//! The downloaded file is cached in OUT_DIR so subsequent builds don't re-download.

use std::env;
#[cfg(feature = "fetch-spec")]
use std::io::Read;
use std::path::PathBuf;

/// Base URL for raw file access to the Khronos Vulkan-Docs repository
#[cfg(feature = "fetch-spec")]
const KHRONOS_RAW_BASE: &str = "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs";

/// Known paths where vk.xml has lived across Vulkan-Docs history.
/// Tried in order — the first successful download wins.
#[cfg(feature = "fetch-spec")]
const VK_XML_REPO_PATHS: &[&str] = &[
    "xml/vk.xml",      // Current layout (roughly v1.2.140+)
    "src/spec/vk.xml", // Older layout (v1.1.70 – ~v1.2.139)
];

/// An error whose `Debug` is its `Display`.
///
/// `fn main() -> Result<_, E>` reports failures with `{:?}`, and a
/// `Box<dyn Error>` built from a String debug-prints it QUOTED AND ESCAPED.
/// Every newline in the messages below reached the reader as a literal escape
/// on one long line, so the two-audience formatting was written, compiled and
/// then discarded at the last step. Nothing failed, because a build script
/// that errors is already failing; only the legibility was lost, and only for
/// the dependent who has nothing else to go on.
struct PlainError(String);

impl std::fmt::Debug for PlainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Display for PlainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PlainError {}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VK_XML_PATH");
    println!("cargo:rerun-if-env-changed=VK_VERSION");

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", timestamp);

    let out_dir = env::var("OUT_DIR")?;
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let output_path = PathBuf::from(&out_dir).join("vulkan_bindings.rs");

    // Resolve vk.xml location
    let xml_path = resolve_vk_xml(&manifest_dir, &out_dir)?;

    // Tell Cargo to re-run if the resolved file changes
    println!("cargo:rerun-if-changed={}", xml_path.display());

    // Generate bindings
    vulkan_gen::generate_bindings(&xml_path, &output_path)?;

    // Export the absolute path of the generated bindings file so
    // integration tests can read it back as text and assert
    // generator-quality invariants (e.g. zero `// TODO:` lines).
    println!(
        "cargo:rustc-env=SPOCK_GENERATED_BINDINGS={}",
        output_path.display()
    );

    println!("Generated Vulkan bindings from {}", xml_path.display());

    Ok(())
}

/// Resolve the path to vk.xml. Four routes, first match wins — see the module
/// docs for which are reachable by a dependent and which are repository-only.
///
/// 1. `VK_XML_PATH` (as given, then relative to this crate's parent)
/// 2. Bundled `<CARGO_MANIFEST_DIR>/vk.xml` — ships in the published crate
/// 3. `../spec/registry/Vulkan-Docs/xml/vk.xml` — this repository only
/// 4. Auto-download, if the `fetch-spec` feature is enabled
fn resolve_vk_xml(
    manifest_dir: &str,
    #[cfg_attr(not(feature = "fetch-spec"), allow(unused))] out_dir: &str,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    // 1. Check VK_XML_PATH environment variable
    if let Ok(env_path) = env::var("VK_XML_PATH") {
        let path = PathBuf::from(&env_path);
        if path.exists() {
            println!("Using vk.xml from VK_XML_PATH: {}", path.display());
            return Ok(path);
        }
        // Named for what it IS rather than what it is in this repo: the base is
        // the parent of CARGO_MANIFEST_DIR, which happens to be the workspace
        // root here and is cargo's registry cache for a dependency.
        let parent_relative = PathBuf::from(manifest_dir).join("..").join(&env_path);
        if parent_relative.exists() {
            println!(
                "Using vk.xml from VK_XML_PATH (relative to this crate's parent): {}",
                parent_relative.display()
            );
            return Ok(parent_relative);
        }
        return Err(PlainError(format!(
            "VK_XML_PATH is set to '{}' but the file does not exist \
             (tried it as given, then relative to this crate's parent directory)",
            env_path
        ))
        .into());
    }

    // 2a. Check bundled copy in the crate directory (ships with the published crate
    //     so docs.rs and other sandboxed builds work without network access).
    let bundled_path = PathBuf::from(manifest_dir).join("vk.xml");
    if bundled_path.exists() {
        println!("Using bundled vk.xml: {}", bundled_path.display());
        return Ok(bundled_path);
    }

    // 2b. Check the repository checkout. Reachable only from a clone of this
    //     repo -- for a dependency, `..` is cargo's registry cache.
    let local_path = PathBuf::from(manifest_dir).join("../spec/registry/Vulkan-Docs/xml/vk.xml");
    if local_path.exists() {
        println!("Using local vk.xml: {}", local_path.display());
        return Ok(local_path);
    }

    // 3. Auto-download if fetch-spec feature is enabled
    #[cfg(feature = "fetch-spec")]
    {
        let version = env::var("VK_VERSION").ok();
        download_vk_xml(out_dir, version.as_deref())
    }

    #[cfg(not(feature = "fetch-spec"))]
    {
        // Two audiences reach this and the routes differ between them. The
        // previous message gave WORKSPACE-DEVELOPER instructions to everyone: a
        // path resolved from CARGO_MANIFEST_DIR (which for a dependency is
        // inside cargo's registry cache) and a `cargo build -p vulkane`
        // invocation (which a dependent never runs). Someone consuming this
        // crate could follow every line of it and get nowhere.
        Err(PlainError(
            "vk.xml not found.\n\
             \n\
             \x20A copy normally ships inside this crate, so reaching this as a\n\
             \x20DEPENDENT means the bundled vk.xml is missing from the package --\n\
             \x20that is unusual and worth reporting. To get moving now, either:\n\
             \x20 1. Set VK_XML_PATH to an ABSOLUTE path. A relative one is resolved\n\
             \x20    against this crate's own directory, then against its parent --\n\
             \x20    inside cargo's registry cache for a dependency, so neither base\n\
             \x20    is anything you control. Or:\n\
             \x20 2. Add the fetch-spec feature to your existing vulkane dependency,\n\
             \x20    which lets the build download one (needs network at build time):\n\
             \x20      features = [\"fetch-spec\"]\n\
             \x20    optionally pinned with VK_VERSION=1.3.250\n\
             \n\
             \x20If you are working IN the vulkane repository: either of the above,\n\
             \x20or place the Vulkan-Docs checkout at spec/registry/Vulkan-Docs/\n\
             \x20(read as ../spec/registry/Vulkan-Docs/xml/vk.xml from this crate\n\
             \x20directory), or build with --features fetch-spec."
                .to_string(),
        )
        .into())
    }
}

/// Download vk.xml from the Khronos GitHub repository.
///
/// If `version` is Some (e.g., "1.3.250"), downloads the tagged release.
/// If `version` is None, downloads from the main branch (latest).
///
/// Caches the download in OUT_DIR to avoid re-downloading on every build.
/// When a specific version is requested, the cache is keyed by version and
/// never expires. For "latest", the cache expires after 24 hours.
#[cfg(feature = "fetch-spec")]
fn download_vk_xml(
    out_dir: &str,
    version: Option<&str>,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let cache_filename = match version {
        Some(v) => format!("vk-{}.xml", v),
        None => "vk.xml".to_string(),
    };
    let cached_path = PathBuf::from(out_dir).join(&cache_filename);

    // Check cache
    if cached_path.exists() {
        match version {
            Some(v) => {
                // Pinned versions never expire
                println!(
                    "Using cached vk.xml for version {}: {}",
                    v,
                    cached_path.display()
                );
                return Ok(cached_path);
            }
            None => {
                // "Latest" cache expires after 24 hours
                if let Ok(metadata) = std::fs::metadata(&cached_path)
                    && let Ok(modified) = metadata.modified()
                {
                    let age = std::time::SystemTime::now()
                        .duration_since(modified)
                        .unwrap_or_default();
                    if age.as_secs() < 86400 {
                        println!(
                            "Using cached vk.xml ({}h old): {}",
                            age.as_secs() / 3600,
                            cached_path.display()
                        );
                        return Ok(cached_path);
                    }
                }
            }
        }
    }

    // Build the git ref for the URL
    let git_ref = match version {
        Some(v) => format!("refs/tags/v{}", v),
        None => "refs/heads/main".to_string(),
    };

    let label = version.unwrap_or("latest");
    println!("Downloading vk.xml ({})...", label);

    // Try each known repo path until one succeeds
    let mut last_error = String::new();
    for repo_path in VK_XML_REPO_PATHS {
        let url = format!("{}/{}/{}", KHRONOS_RAW_BASE, git_ref, repo_path);
        println!("  Trying {}...", url);

        match ureq::get(&url).call() {
            Ok(response) => {
                let mut content = Vec::new();
                response.into_body().as_reader().read_to_end(&mut content)?;

                let content_str = std::str::from_utf8(&content)?;
                if !content_str.contains("<registry>") {
                    last_error = format!("{} returned non-XML content", url);
                    continue;
                }

                std::fs::write(&cached_path, &content)?;

                // Log the version we got
                if let Some(ver_line) = content_str.lines().find(|l| {
                    l.contains("VK_HEADER_VERSION")
                        && !l.contains("COMPLETE")
                        && l.contains("#define")
                }) {
                    println!("  {}", ver_line.trim());
                }

                println!("  Cached to {}", cached_path.display());
                return Ok(cached_path);
            }
            Err(e) => {
                last_error = format!("{}: {}", url, e);
                continue;
            }
        }
    }

    Err(format!(
        "Failed to download vk.xml for version '{}'. \
         Tried all known repository paths.\n  Last error: {}",
        label, last_error
    )
    .into())
}
