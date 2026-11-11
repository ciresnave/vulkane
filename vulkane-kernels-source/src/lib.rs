// SPDX-License-Identifier: MIT OR Apache-2.0
//! # vulkane-kernels-source
//!
//! Home for the GLSL GPU kernel sources (`.glsl`) and the `compile.sh`
//! script that turns them into SPIR-V. Moved from Fuel's `fuel-kernels-source`
//! with its git history; see `README.md` and `THIRD-PARTY-NOTICES.md`. The Slang
//! kernels moved to FishStringer's `fishstringer-kernels-source` (CireSnave, 2026-11-06).
//!
//! This crate intentionally has no Rust code. Consumers read the sources from
//! the published package (`kernels/`) and build or embed SPIR-V themselves.
//!
//! ## Layout
//!
//! - `kernels/*.glsl` — GLSL sources, kept where GLSL is more ergonomic
//!   (e.g. cooperative-matrix matmul).
//! - `kernels/compile.sh [--out-dir DIR] [files...]` — ahead-of-time compiler
//!   invocation; SPIR-V goes to `DIR` (default: `spv/` next to `kernels/`).
