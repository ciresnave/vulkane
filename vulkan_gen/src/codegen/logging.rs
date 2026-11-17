// SPDX-License-Identifier: MIT OR Apache-2.0
//! Simple logging utilities for the code generator

/// Print an `[INFO]` line to stderr.
pub fn log_info(msg: &str) {
    eprintln!("[INFO] {}", msg);
}

/// Debug messages are suppressed in normal builds; this does nothing.
pub fn log_debug(_msg: &str) {
    // Debug messages are suppressed in normal builds
}

/// Print a `[WARN]` line to stderr.
pub fn log_warn(msg: &str) {
    eprintln!("[WARN] {}", msg);
}

/// Print an `[ERROR]` line to stderr.
pub fn log_error(msg: &str) {
    eprintln!("[ERROR] {}", msg);
}
