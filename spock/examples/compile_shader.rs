//! One-shot helper that compiles `shaders/square_buffer.comp` to SPIR-V
//! using the optional `naga` feature, and writes the result alongside the
//! source file.
//!
//! Run with: `cargo run -p spock --features naga --example compile_shader`
//!
//! The generated `.spv` file is checked into the repository, so users running
//! the `compute_square` example don't need `naga`, `glslc`, or any shader
//! toolchain. This helper is only needed if you edit the GLSL source.

#[cfg(not(feature = "naga"))]
fn main() {
    eprintln!(
        "This example requires the `naga` feature.\n\
         Run with: cargo run -p spock --features naga --example compile_shader"
    );
    std::process::exit(1);
}

#[cfg(feature = "naga")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use spock::safe::naga::compile_glsl;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let glsl_path = format!("{manifest_dir}/examples/shaders/square_buffer.comp");
    let spv_path = format!("{manifest_dir}/examples/shaders/square_buffer.spv");

    println!("Reading {glsl_path}");
    let source = std::fs::read_to_string(&glsl_path)?;

    println!("Compiling GLSL -> SPIR-V via naga");
    let words = compile_glsl(&source, ::naga::ShaderStage::Compute)?;

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

    println!("Done.");
    Ok(())
}
