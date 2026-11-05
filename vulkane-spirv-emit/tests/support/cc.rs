// SPDX-License-Identifier: MIT OR Apache-2.0
// Copied verbatim from `unpopped-cpu-c` 0.43.2 `tests/support/cc.rs` (same author, MIT OR Apache-2.0; test-only, not
// packaged: Cargo.toml `include` omits `tests/`). Only this note and the `allow` below were added.
#![allow(dead_code)]
//! Host C compiler discovery and driving, shared by the compile-and-run tests
//! (moved verbatim from `cpu_end_to_end.rs` so the kernel-layer differential can use it).

use std::path::{Path, PathBuf};
use std::process::Command;

/// A host C compiler and how to drive it.
pub(crate) struct CCompiler {
    /// Base command, already carrying whatever environment it needs (the MSVC
    /// tool from `cc`'s registry lookup carries INCLUDE/LIB; a PATH `cc` does not
    /// need any).
    cmd: &'static str,
    /// `cl.exe`-style flags (`-Fe:`) rather than GNU-style (`-o`).
    msvc: bool,
    /// Whether the link line needs `-lm`.
    ///
    /// This is *not* implied by the driver being GNU-style. Visual Studio ships
    /// a `clang` that targets `x86_64-pc-windows-msvc`, where the math functions
    /// live in the UCRT and there is no `m.lib` at all — passing `-lm` there is
    /// a hard `lld-link: could not open 'm.lib'`. So ask the driver what it
    /// targets instead of assuming from its name.
    needs_libm: bool,
    /// A `vcvars64.bat` to `call` before the compiler, for the normal Windows
    /// case where Visual Studio is installed but deliberately not on `PATH`.
    vcvars: Option<PathBuf>,
}

/// Locate a host C compiler, or `None`.
///
/// A plain `PATH` lookup first, then — on Windows — a **Visual Studio install
/// that is not on `PATH`**, which is the normal state of a Windows box rather
/// than an unusual one: the installer does not modify the global environment,
/// and `vcvars64.bat` is the supported way in.
///
/// Skipping that fallback is what made this whole file silently vacuous here.
/// `find_compiler` returned `None`, every e2e test printed `SKIP` and passed,
/// and the suite reported green while compiling nothing — on a machine with a
/// perfectly good toolchain sitting in Program Files. A test that cannot run
/// where it is written is not a test there, and one that says `ok` while doing
/// so is worse than one that fails.
pub(crate) fn find_compiler() -> Option<CCompiler> {
    for (cmd, msvc) in [
        ("cc", false),
        ("gcc", false),
        ("clang", false),
        ("cl", true),
    ] {
        if msvc {
            // `cl` with no arguments prints its banner and exits non-zero; a
            // successful *spawn* is the signal that it is on PATH.
            if Command::new(cmd).output().is_ok() {
                return Some(CCompiler {
                    cmd,
                    msvc,
                    needs_libm: false,
                    vcvars: None,
                });
            }
            continue;
        }
        if Command::new(cmd).arg("--version").output().is_err() {
            continue;
        }
        // `-dumpmachine` is understood by gcc, clang and any `cc` symlinked to
        // either. If it fails, fall back to needing libm — the Unix default.
        let triple = Command::new(cmd)
            .arg("-dumpmachine")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let needs_libm = !triple.contains("windows-msvc");
        return Some(CCompiler {
            cmd,
            msvc,
            needs_libm,
            vcvars: None,
        });
    }
    let found = find_vs_install().map(|vcvars| CCompiler {
        cmd: "cl",
        msvc: true,
        needs_libm: false,
        vcvars: Some(vcvars),
    });

    // UNDER CI, A MISSING COMPILER IS A FAILURE RATHER THAN A SKIP.
    //
    // The file header records the decision to skip loudly where there is
    // genuinely no toolchain, and that stays right for a developer's machine —
    // nobody should be blocked from running the rest of the suite by a missing
    // `cc`. But it leaves a hole this file's own words already name: **a loud
    // skip is still a pass**, the `eprintln!` is CAPTURED by the harness on a
    // passing test, and the run reports `ok` either way. So in a CI log a skip
    // and a real execution are indistinguishable, and the strongest gate this
    // workspace has could stop running with nothing to show for it.
    //
    // A runner always has a toolchain. `CI` is set by GitHub Actions and by
    // every other runner worth naming, so where it is set a `None` here is not
    // "no toolchain available" — it is "the toolchain we were promised is
    // missing", which is a fact worth failing on.
    if found.is_none() && std::env::var_os("CI").is_some() {
        panic!(
            "no host C compiler under CI (tried cc, gcc, clang, cl, and the              vswhere/vcvars fallback). Every real-execution test in this file              would SKIP and report `ok`, so the suite would go green while              compiling nothing. Under CI that is a failure, not a skip."
        );
    }
    found
}

impl CCompiler {
    /// Compile `src` to `exe`. Returns the compiler's diagnostics on failure.
    ///
    /// `optimize` selects `-O2` / `/O2` instead of the unoptimized default. It
    /// matters for exactly one caller and it is not a performance knob: the
    /// NaN-propagation test is checking whether an **optimizer** contracts the
    /// emitted ternary into a NaN-suppressing max instruction. Run at `-O0` that
    /// test passes trivially and proves nothing, because there is no optimizer to
    /// do the contracting. The value-correctness tests stay unoptimized so a
    /// failure there is unambiguously the emitter's rather than the compiler's.
    pub(crate) fn compile(&self, src: &Path, exe: &Path, optimize: bool) -> Result<(), String> {
        // A `vcvars` install has to be entered through `cmd`, because the batch
        // file's whole job is to mutate the environment of the process that
        // calls it. Spawning `cl` directly after running it in a *different*
        // process would inherit nothing.
        if let Some(vcvars) = &self.vcvars {
            let opt = if optimize { "/O2" } else { "/Od" };
            // Driven through a generated .bat rather than `cmd /C "<line>"`.
            // Rust escapes arguments for `CreateProcess`, `cmd` then applies its
            // own quoting rules to what arrives, and a command line carrying
            // three quoted Windows paths does not survive both. The failure mode
            // is silent: `cmd` exits non-zero having printed nothing at all, so
            // the compile looks like it failed rather than like it never ran.
            // A batch file has exactly one quoting layer.
            let bat = src.with_extension("build.bat");
            let script = format!(
                "@echo off\r\ncall \"{}\" >nul 2>&1\r\ncl /nologo {opt} \"{}\" /Fe:\"{}\"\r\n",
                vcvars.display(),
                src.display(),
                exe.display()
            );
            std::fs::write(&bat, script).map_err(|e| format!("write build script: {e}"))?;
            let out = Command::new(&bat)
                .current_dir(src.parent().unwrap_or(Path::new(".")))
                .output()
                .map_err(|e| format!("spawn build script failed: {e}"))?;
            if out.status.success() {
                return Ok(());
            }
            return Err(format!(
                "compile failed (via vcvars)\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        let mut c = Command::new(self.cmd);
        if self.msvc {
            c.arg("-nologo")
                .arg(if optimize { "-O2" } else { "-Od" })
                .arg(src)
                .arg(format!("-Fe:{}", exe.display()));
        } else {
            c.arg(src)
                .arg("-o")
                .arg(exe)
                .arg(if optimize { "-O2" } else { "-O0" });
            if self.needs_libm {
                c.arg("-lm");
            }
        }
        let out = c.output().map_err(|e| format!("spawn failed: {e}"))?;
        if out.status.success() {
            return Ok(());
        }
        Err(format!(
            "compile failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

/// The `vcvars64.bat` of the newest Visual Studio install carrying the C/C++
/// tools, or `None`.
///
/// `vswhere.exe` is Microsoft's supported discovery mechanism and ships at a
/// fixed location, which is what makes this a lookup rather than a guess at
/// version-numbered directory names.
#[cfg(windows)]
fn find_vs_install() -> Option<PathBuf> {
    let pf86 = std::env::var("ProgramFiles(x86)")
        .unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());
    let vswhere = PathBuf::from(pf86)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    if !vswhere.exists() {
        return None;
    }
    let out = Command::new(&vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if root.is_empty() {
        return None;
    }
    let bat = PathBuf::from(root)
        .join("VC")
        .join("Auxiliary")
        .join("Build")
        .join("vcvars64.bat");
    bat.exists().then_some(bat)
}

#[cfg(not(windows))]
fn find_vs_install() -> Option<PathBuf> {
    None
}
