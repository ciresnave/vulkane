// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unit tests for the vulkane crate's own modules (version, loader)

#[test]
fn test_version_module() {
    use crate::version::*;

    assert!(!BINDINGS_VERSION.is_empty());
    assert!(!BUILD_TIMESTAMP.is_empty());
}

#[test]
fn test_version_struct_roundtrip() {
    use crate::raw::version::Version;

    let v = Version::new(1, 3, 42);
    let raw = v.to_raw();
    let decoded = Version::from_raw(raw);
    assert_eq!(decoded.major, 1);
    assert_eq!(decoded.minor, 3);
    assert_eq!(decoded.patch, 42);
}

#[test]
fn test_bindings_reexport() {
    // The crate root re-exports raw::bindings::* so these should be accessible
    let _ = crate::VK_API_VERSION_1_0;
}
