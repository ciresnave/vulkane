// SPDX-License-Identifier: MIT OR Apache-2.0
//! # vulkane-kernels-source
//!
//! Home for GPU kernel source files (`.slang`, `.glsl`) and the `compile.sh`
//! script that turns them into SPIR-V. Moved from Fuel's `fuel-kernels-source`
//! with its git history; see `README.md` and `THIRD-PARTY-NOTICES.md`.
//!
//! This crate intentionally has no Rust code. Consumers read the sources from
//! the published package (`kernels/`) and build or embed SPIR-V themselves.
//!
//! ## Layout
//!
//! - `kernels/*.slang` — Slang sources (preferred for new kernels:
//!   autodiff-ready, cross-backend output).
//! - `kernels/*.glsl` — GLSL sources, kept where GLSL is more ergonomic
//!   (e.g. cooperative-matrix matmul).
//! - `kernels/compile.sh [--out-dir DIR] [files...]` — ahead-of-time compiler
//!   invocation; SPIR-V goes to `DIR` (default: `spv/` next to `kernels/`).
//! - `kernels/hoist_extinst_imports.py` — post-pass that hoists
//!   `OpExtInstImport` out of function bodies in `slangc` output.
