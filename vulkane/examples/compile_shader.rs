// SPDX-License-Identifier: MIT OR Apache-2.0
//! One-shot helper that compiles every GLSL file under
//! `examples/shaders/` to its matching `*.spv` SPIR-V binary using the
//! optional `naga` feature. The extension determines the shader stage:
//! `.comp` -> compute, `.vert` -> vertex, `.frag` -> fragment.
//!
//! Run with: `cargo run -p vulkane --features naga,fetch-spec --example compile_shader`
//!
//! The generated `.spv` files are checked into the repository, so users
//! running the bundled examples don't need `naga`, `glslc`, or any
//! shader toolchain. This helper is only needed if you edit the GLSL
//! sources.

#[cfg(not(feature = "naga"))]
fn main() {
    eprintln!(
        "This example requires the `naga` feature.\n\
         Run with: cargo run -p vulkane --features naga,fetch-spec --example compile_shader"
    );
    std::process::exit(1);
}

#[cfg(feature = "naga")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use vulkane::safe::naga::{compile_glsl, compile_wgsl};

    enum SourceKind {
        Glsl(::naga::ShaderStage),
        Wgsl,
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let shaders_dir = format!("{manifest_dir}/examples/shaders");

    // Compile every supported shader file in the shaders directory.
    let mut found = 0usize;
    for entry in std::fs::read_dir(&shaders_dir)? {
        let entry = entry?;
        let path = entry.path();
        let kind = match path.extension().and_then(|s| s.to_str()) {
            Some("comp") => SourceKind::Glsl(::naga::ShaderStage::Compute),
            Some("vert") => SourceKind::Glsl(::naga::ShaderStage::Vertex),
            Some("frag") => SourceKind::Glsl(::naga::ShaderStage::Fragment),
            // WGSL modules carry every entry point in one file. The
            // resulting .spv has all of them; the consumer picks which
            // entry to run via the pipeline stage's `pName`.
            Some("wgsl") => SourceKind::Wgsl,
            _ => continue,
        };
        found += 1;
        let src_path = path.display().to_string();
        // For .comp shaders we keep the legacy `.spv` suffix (the
        // existing compute examples already reference those names).
        // For .vert / .frag we suffix as `.vert.spv` / `.frag.spv` so
        // a single `triangle.vert` and `triangle.frag` pair don't
        // collide on `triangle.spv`. WGSL gets `.wgsl.spv`.
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("shader");
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let spv_filename = if ext == "comp" {
            format!("{stem}.spv")
        } else {
            format!("{stem}.{ext}.spv")
        };
        let spv_path = path.with_file_name(&spv_filename).display().to_string();

        println!("Reading {src_path}");
        let source = std::fs::read_to_string(&path)?;

        let words = match kind {
            SourceKind::Glsl(stage) => {
                println!("Compiling GLSL -> SPIR-V via naga ({stage:?})");
                compile_glsl(&source, stage)?
            }
            SourceKind::Wgsl => {
                println!("Compiling WGSL -> SPIR-V via naga");
                compile_wgsl(&source)?
            }
        };

        // Write as little-endian bytes (the universal SPIR-V on-disk format).
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        println!(
            "Writing {} bytes ({} u32 words) to {spv_path}",
            bytes.len(),
            words.len()
        );
        std::fs::write(&spv_path, &bytes)?;
    }

    if found == 0 {
        eprintln!("No shader sources found under {shaders_dir}");
    } else {
        println!("Done. Compiled {found} shader(s).");
    }
    Ok(())
}
