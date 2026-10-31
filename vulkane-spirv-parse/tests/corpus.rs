// SPDX-License-Identifier: MIT OR Apache-2.0
//! Round-trip and reflection over real modules.
//!
//! The 9 `.spv` files committed under `vulkane/examples/shaders` always run. Set
//! `VULKANE_SPV_CORPUS` to a directory of `.spv` files (the 178 kernels built by
//! `vulkane-kernels-source/compile.sh`, or Fuel's `fuel-vulkan-kernels/spv`) to run them all;
//! if it is set but yields no module the test fails rather than passing on nothing.

use std::path::{Path, PathBuf};

use vulkane_spirv_parse::Module;

#[allow(clippy::chunks_exact_to_as_chunks)]
fn words(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0, "{} is not word aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn spv_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "spv"))
        .collect();
    v.sort();
    v
}

fn committed() -> Vec<PathBuf> {
    spv_in(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../vulkane/examples/shaders"))
}

fn round_trip(paths: &[PathBuf]) {
    for p in paths {
        let w = words(p);
        let m = Module::parse(&w).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let out = m.to_words();
        let first = w.iter().zip(&out).position(|(a, b)| a != b);
        assert!(
            first.is_none() && out.len() == w.len(),
            "{} does not round-trip: {} -> {} words, first difference at word {first:?}",
            p.display(),
            w.len(),
            out.len()
        );
    }
}

#[test]
fn committed_modules_round_trip() {
    let paths = committed();
    assert_eq!(paths.len(), 9, "expected the 9 committed modules");
    round_trip(&paths);
}

#[test]
fn compute_module_reflection() {
    let m = Module::parse(&words(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../vulkane/examples/shaders/square_buffer.spv"),
    ))
    .unwrap();
    let eps = m.entry_points();
    assert_eq!(eps.len(), 1);
    assert_eq!(format!("{:?}", eps[0].model), "GLCompute");
    assert!(
        eps[0].local_size.is_some(),
        "compute entry needs a LocalSize"
    );
    assert!(
        !m.bindings().is_empty(),
        "compute kernel has buffer bindings"
    );
    assert!(!m.capabilities().is_empty());
}

#[test]
fn truncated_module_is_rejected() {
    let w = words(&committed()[0]);
    assert!(
        Module::parse(&w[..w.len() - 1]).is_err()
            || Module::parse(&w[..w.len() - 1]).unwrap().to_words() != w
    );
}

#[test]
fn external_corpus_round_trips() {
    let Ok(dir) = std::env::var("VULKANE_SPV_CORPUS") else {
        return;
    };
    let paths = spv_in(Path::new(&dir));
    assert!(
        !paths.is_empty(),
        "VULKANE_SPV_CORPUS={dir} holds no .spv files"
    );
    round_trip(&paths);
    eprintln!("round-tripped {} modules from {dir}", paths.len());
}
