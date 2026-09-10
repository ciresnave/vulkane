// SPDX-License-Identifier: MIT OR Apache-2.0
//! Utilities for emitting provisional (stub) types during code generation.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// The kind of provisional type to emit.
pub enum ProvisionalKind {
    Struct,
    Handle,
    Enum,
    Alias,
    Basic,
}

/// Emit a provisional type stub to the provisional_types.rs file.
pub fn emit_provisional_type(name: &str, kind: ProvisionalKind, out_dir: &Path) {
    let file_path = out_dir.join("provisional_types.rs");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)
        .expect("Failed to open provisional_types.rs");
    let code = match kind {
        ProvisionalKind::Struct => format!(
            "// PROVISIONAL: auto-generated stub for missing struct\npub struct {} {{ _private: () }}\n\n",
            name
        ),
        ProvisionalKind::Handle => format!(
            "// PROVISIONAL: auto-generated stub for missing handle\npub type {} = *mut std::ffi::c_void;\n\n",
            name
        ),
        ProvisionalKind::Enum => format!(
            "// PROVISIONAL: auto-generated stub for missing enum\npub enum {} {{ _Dummy }}\n\n",
            name
        ),
        ProvisionalKind::Alias => format!(
            "// PROVISIONAL: auto-generated stub for missing alias\npub type {} = ();\n\n",
            name
        ),
        ProvisionalKind::Basic => format!(
            "// PROVISIONAL: auto-generated stub for missing basic type\npub type {} = ();\n\n",
            name
        ),
    };
    file.write_all(code.as_bytes())
        .expect("Failed to write provisional type");
}
