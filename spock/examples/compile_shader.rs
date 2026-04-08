//! One-shot helper that compiles every GLSL file under
//! `examples/shaders/` to its matching `*.spv` SPIR-V binary using the
//! optional `naga` feature. The extension determines the shader stage:
//! `.comp` -> compute, `.vert` -> vertex, `.frag` -> fragment.
//!
//! Run with: `cargo run -p spock --features naga,fetch-spec --example compile_shader`
//!
//! The generated `.spv` files are checked into the repository, so users
//! running the bundled examples don't need `naga`, `glslc`, or any
//! shader toolchain. This helper is only needed if you edit the GLSL
//! sources.

#[cfg(not(feature = "naga"))]
fn main() {
    eprintln!(
        "This example requires the `naga` feature.\n\
         Run with: cargo run -p spock --features naga,fetch-spec --example compile_shader"
    );
    std::process::exit(1);
}

#[cfg(feature = "naga")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use spock::safe::naga::compile_glsl;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let shaders_dir = format!("{manifest_dir}/examples/shaders");

    // Compile every supported GLSL file in the shaders directory.
    let mut found = 0usize;
    for entry in std::fs::read_dir(&shaders_dir)? {
        let entry = entry?;
        let path = entry.path();
        let stage = match path.extension().and_then(|s| s.to_str()) {
            Some("comp") => ::naga::ShaderStage::Compute,
            Some("vert") => ::naga::ShaderStage::Vertex,
            Some("frag") => ::naga::ShaderStage::Fragment,
            _ => continue,
        };
        found += 1;
        let glsl_path = path.display().to_string();
        // For .comp shaders we keep the legacy `.spv` suffix (the
        // existing compute examples already reference those names).
        // For .vert / .frag we suffix as `.vert.spv` / `.frag.spv` so
        // a single `triangle.vert` and `triangle.frag` pair don't
        // collide on `triangle.spv`.
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

        println!("Reading {glsl_path}");
        let source = std::fs::read_to_string(&path)?;

        println!("Compiling GLSL -> SPIR-V via naga ({stage:?})");
        let words = compile_glsl(&source, stage)?;

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
        eprintln!("No GLSL shader files found under {shaders_dir}");
    } else {
        println!("Done. Compiled {found} shader(s).");
    }
    Ok(())
}
