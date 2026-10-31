// SPDX-License-Identifier: MIT OR Apache-2.0
//! `spirv-val` verdicts before and after a parse/encode, and the "parser does not launder invalid
//! modules" case (vulkane#106, ROADMAP 3a).
//!
//! Needs SPIR-V Tools' `spirv-val` on `PATH`. Without it each test returns early with a note,
//! unless `VULKANE_REQUIRE_SPIRV_VAL` is set (CI sets it), in which case a missing tool fails.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rspirv::binary::Assemble;
use vulkane_spirv_parse::Module;

const TARGET_ENV: &str = "vulkan1.3";

fn words(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn committed() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vulkane/examples/shaders");
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "spv"))
        .collect();
    v.sort();
    v
}

/// `Some(accepted)`, or `None` when `spirv-val` is unavailable and not required.
///
/// The module goes over stdin (`-`), so no temporary file is created.
fn validates(w: &[u32]) -> Option<bool> {
    let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let out = Command::new("spirv-val")
        .args(["--target-env", TARGET_ENV, "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            // spirv-val may reject and exit before reading everything: a broken pipe is not an error here.
            let _ = child.stdin.take().expect("piped stdin").write_all(&bytes);
            child.wait_with_output()
        });
    match out {
        Ok(o) => Some(o.status.success()),
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_SPIRV_VAL").is_none(),
                "VULKANE_REQUIRE_SPIRV_VAL is set but `spirv-val` could not be run: {e}"
            );
            eprintln!("spirv-val unavailable ({e}); skipping");
            None
        }
    }
}

#[test]
fn verdict_survives_parse_and_encode() {
    let mut accepted = 0;
    for p in committed() {
        let w = words(&p);
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let Some(before) = validates(&w) else {
            return;
        };
        let after = validates(&Module::parse(&w).unwrap().to_words()).unwrap();
        assert_eq!(
            before, after,
            "{name}: verdict changed through parse/encode"
        );
        accepted += usize::from(before);
    }
    // The comparison is vacuous if everything is rejected both ways.
    assert!(
        accepted > 0,
        "no committed module passes spirv-val {TARGET_ENV}"
    );
}

#[test]
fn parse_does_not_launder_an_invalid_module() {
    let w = words(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../vulkane/examples/shaders/square_buffer.spv"),
    );
    // Positive control: the unmodified module is valid.
    let Some(valid) = validates(&w) else {
        return;
    };
    assert!(valid, "control module must pass spirv-val {TARGET_ENV}");

    // Well-formed but invalid: no capabilities declared, so `Shader` features are used undeclared.
    let mut inner = Module::parse(&w).unwrap().as_rspirv().clone();
    inner.capabilities.clear();
    let broken = inner.assemble();
    let reparsed = Module::parse(&broken).expect("a well-formed binary must still parse");
    assert_eq!(
        reparsed.to_words(),
        broken,
        "parser must not repair the module"
    );
    assert_eq!(validates(&broken), Some(false), "spirv-val must reject it");
}
