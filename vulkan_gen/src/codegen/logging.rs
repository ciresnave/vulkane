// SPDX-License-Identifier: MIT OR Apache-2.0
//! Simple logging utilities for the code generator

pub fn log_info(msg: &str) {
    eprintln!("[INFO] {}", msg);
}

pub fn log_debug(_msg: &str) {
    // Debug messages are suppressed in normal builds
}

pub fn log_warn(msg: &str) {
    eprintln!("[WARN] {}", msg);
}

pub fn log_error(msg: &str) {
    eprintln!("[ERROR] {}", msg);
}
