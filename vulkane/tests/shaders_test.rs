//! Tests for the shader registry (embedded + env-override lookup).
//!
//! Does not exercise `load_module`, which needs a live `Device` — that
//! path is trivial glue on top of `load` and is covered indirectly by
//! downstream consumers.

use std::fs;
use std::path::PathBuf;

use vulkane::safe::{ShaderLoadError, ShaderRegistry, ShaderSource};

// SPIR-V magic number as little-endian bytes. A 4-byte "shader" made
// just of the magic number passes the length-multiple-of-4 check and
// lets us verify that load_words produces the expected word.
const SPIRV_MAGIC_LE: [u8; 4] = [0x03, 0x02, 0x23, 0x07];

static TRIVIAL_SHADER: ShaderSource = ShaderSource {
    name: "trivial",
    spv: &SPIRV_MAGIC_LE,
};

static ODD_SHADER: ShaderSource = ShaderSource {
    name: "odd_length",
    spv: &[0x03, 0x02, 0x23], // 3 bytes, not a multiple of 4
};

static EMBEDDED: &[ShaderSource] = &[TRIVIAL_SHADER, ODD_SHADER];

fn registry() -> ShaderRegistry {
    ShaderRegistry::new().with_embedded(EMBEDDED)
}

/// Create a per-test temp directory (process-id + test-name suffixed so
/// parallel tests don't collide) and return its path. Caller is
/// responsible for cleanup.
fn make_temp_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vulkane_shaders_test_{}_{}",
        std::process::id(),
        test_name,
    ));
    let _ = fs::remove_dir_all(&dir); // ignore if absent
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn load_embedded_returns_exact_bytes() {
    let reg = registry();
    let bytes = reg.load("trivial").expect("embedded shader loads");
    assert_eq!(&*bytes, &SPIRV_MAGIC_LE);
}

#[test]
fn load_missing_returns_not_found() {
    let reg = registry();
    let err = reg.load("does_not_exist").unwrap_err();
    assert!(matches!(err, ShaderLoadError::NotFound(ref n) if n == "does_not_exist"));
}

#[test]
fn load_words_decodes_little_endian() {
    let reg = registry();
    let words = reg.load_words("trivial").expect("word decode succeeds");
    assert_eq!(words, vec![0x07230203]);
}

#[test]
fn load_words_rejects_non_multiple_of_four() {
    let reg = registry();
    let err = reg.load_words("odd_length").unwrap_err();
    assert!(matches!(
        err,
        ShaderLoadError::MalformedSpirv {
            ref name,
            byte_len: 3,
        } if name == "odd_length"
    ));
}

#[test]
fn env_override_wins_over_embedded() {
    let dir = make_temp_dir("env_override_wins");
    // Custom bytes — different from the embedded version of "trivial".
    let custom: [u8; 8] = [0x03, 0x02, 0x23, 0x07, 0x42, 0x00, 0x00, 0x00];
    fs::write(dir.join("trivial.spv"), custom).expect("write override file");

    // Unique env var per test to avoid cross-test races.
    let var = "VULKANE_TEST_OVERRIDE_env_override_wins";
    // SAFETY: each test uses a unique env-var name, so there are no
    // concurrent writers to the same variable.
    unsafe {
        std::env::set_var(var, &dir);
    }

    let reg = ShaderRegistry::new()
        .with_embedded(EMBEDDED)
        .with_env_override("VULKANE_TEST_OVERRIDE_env_override_wins");

    let bytes = reg.load("trivial").expect("override loads");
    assert_eq!(&*bytes, &custom);

    unsafe {
        std::env::remove_var(var);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn env_override_falls_through_when_file_missing() {
    // Override dir exists but does not contain a file for the shader
    // we're looking up — should fall back to the embedded table.
    let dir = make_temp_dir("falls_through");

    let var = "VULKANE_TEST_OVERRIDE_falls_through";
    unsafe {
        std::env::set_var(var, &dir);
    }

    let reg = ShaderRegistry::new()
        .with_embedded(EMBEDDED)
        .with_env_override("VULKANE_TEST_OVERRIDE_falls_through");

    let bytes = reg.load("trivial").expect("falls through to embedded");
    assert_eq!(&*bytes, &SPIRV_MAGIC_LE);

    unsafe {
        std::env::remove_var(var);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn env_override_ignored_when_var_unset() {
    let reg = ShaderRegistry::new()
        .with_embedded(EMBEDDED)
        .with_env_override("VULKANE_TEST_OVERRIDE_definitely_unset");

    let bytes = reg.load("trivial").expect("loads from embedded");
    assert_eq!(&*bytes, &SPIRV_MAGIC_LE);
}

#[test]
fn empty_registry_returns_not_found_for_everything() {
    let reg = ShaderRegistry::new();
    assert!(matches!(
        reg.load("anything").unwrap_err(),
        ShaderLoadError::NotFound(_)
    ));
}
