// SPDX-License-Identifier: MIT OR Apache-2.0
//! G4.2, the behaviour oracle (`docs/design/spirv-parser-emitter.md` section S4): the same
//! `OpDef` is lowered by `unpopped-cpu-c` (portable C, compiled and run on the host) and by this
//! emitter (SPIR-V, run on a live device) over the same input bits, and the two result
//! bit-streams are compared cell by cell.
//!
//! Integer ops, comparisons, `min`/`max` and the rounding family must agree bit for bit.
//! Transcendentals are not required to: Vulkan grants them a precision budget and a host libm
//! another, so they are held to a tolerance that is small next to any wrong-function error.
//! Both NaNs count as equal (payloads are not specified across the two).
//!
//! `f64` is not run: a device needs `shaderFloat64` enabled, which the shared test device does not
//! request. The validity of the `f64` modules is covered by `spirv-val` in the unit tests.
//!
//! Without a device, or without a host C compiler, the test follows `VULKANE_REQUIRE_DEVICE`
//! (device) and `CI` (compiler, see `support/cc.rs`) rather than passing silently.
//!
//! The second test, `slang_and_spirv_emitters_agree_on_a_device`, is the Slang-vs-SPIR-V cross-check:
//! the same cases through `vulkane-slang-emit` (compiled by `slangc`) and this emitter, both on the
//! device. The cases, tolerances and domains live in `support/cases.rs`.

#[path = "support/cases.rs"]
mod cases;
#[path = "support/cc.rs"]
mod cc;
#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;
#[path = "support/host_c.rs"]
mod host_c;

use std::process::Command;

use unpopped::try_generate;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;
use vulkane_spirv_emit::SpirV;

use cases::{Case, Tol, agree, cases as all_cases};

/// A kernel ready to run on the device: its entry-point name and SPIR-V words.
type Module = (String, Vec<u32>);

/// Disagreements found so far, split by whether the op is specified on that input row.
#[derive(Default)]
struct Tally {
    /// In-domain cells compared.
    compared: usize,
    /// In-domain disagreements: these fail the test.
    bad: Vec<String>,
    /// Out-of-domain disagreements: reported only.
    undefined: Vec<String>,
}

impl Tally {
    /// Check row `j` of `c` against each `(what, got, want)` pair.
    fn cell(&mut self, c: &Case, j: usize, checks: &[(&str, u32, u32)]) {
        let row: Vec<u32> = c.inputs.iter().map(|col| col[j]).collect();
        let in_domain = c.domain.holds(&row);
        self.compared += usize::from(in_domain);
        for &(what, got, want) in checks {
            if agree(c.tol, c.dt, got, want) {
                continue;
            }
            let line = format!(
                "{} {what} ({:?}) inputs {row:x?}: {got:#010x} vs {want:#010x}",
                c.name, c.tol
            );
            if in_domain {
                self.bad.push(line);
            } else {
                self.undefined.push(line);
            }
        }
    }

    /// Print the out-of-domain report and fail on any in-domain disagreement.
    fn finish(&self, tag: &str, against: &str, kernels: usize) {
        eprintln!(
            "{tag}: {} cells outside the ops' specified domains disagree (reported, not failed):\n{}",
            self.undefined.len(),
            self.undefined.join("\n")
        );
        assert!(
            self.bad.is_empty(),
            "{} of {} in-domain cells disagree with {against}:\n{}",
            self.bad.len(),
            self.compared,
            self.bad.join("\n")
        );
        eprintln!(
            "{tag}: {} in-domain cells over {kernels} kernels agree with {against}",
            self.compared
        );
    }
}

/// Each case's SPIR-V module and its `unpopped-cpu-c` kernel `(name, C source)`.
fn generate(cases: &[Case]) -> (Vec<Module>, Vec<(String, String)>) {
    let (mut spirv, mut c_sources) = (Vec::new(), Vec::new());
    for c in cases {
        let s = try_generate(&c.def, &c.key, &SpirV)
            .unwrap_or_else(|e| panic!("{}: the SPIR-V backend declined: {e:?}", c.name));
        spirv.push((s.name.clone(), s.words().expect("SPIR-V words").to_vec()));
        let k = try_generate(&c.def, &c.key, &CpuC)
            .unwrap_or_else(|e| panic!("{}: the CPU-C backend declined: {e:?}", c.name));
        c_sources.push((k.name.clone(), k.source.clone()));
    }
    (spirv, c_sources)
}

fn columns(c: &Case) -> Vec<&[u32]> {
    c.inputs.iter().map(Vec::as_slice).collect()
}

#[test]
fn agree_has_a_control_that_can_fail() {
    let one = 1.0f32.to_bits();
    let next = f32::from_bits(one + 1).to_bits();
    let close = Tol::Close {
        rel: 1e-5,
        abs: 1e-5,
    };
    assert!(!agree(Tol::Exact, ElementKind::F32, one, next));
    assert!(agree(close, ElementKind::F32, one, next));
    assert!(!agree(close, ElementKind::F32, one, 2.0f32.to_bits()));
    assert!(!agree(Tol::Exact, ElementKind::I32, 1, 2));
    assert!(agree(
        Tol::Exact,
        ElementKind::F32,
        f32::NAN.to_bits(),
        (-f32::NAN).to_bits()
    ));
    assert!(agree(
        Tol::ZeroSign,
        ElementKind::F32,
        0.0f32.to_bits(),
        (-0.0f32).to_bits()
    ));
    assert!(!agree(Tol::ZeroSign, ElementKind::F32, one, next));
}

#[test]
fn emitted_spirv_matches_the_cpu_c_oracle_on_a_device() {
    common::require_serialization_lock();
    let Some(cc) = cc::find_compiler() else {
        eprintln!("SKIP: no host C compiler (tried cc, gcc, clang, cl and the vcvars fallback)");
        return;
    };
    let Some(gpu) = gpu::Gpu::open("cpu_c_oracle") else {
        return; // skip already declared
    };
    let cases = all_cases();
    let (spirv, c_sources) = generate(&cases);
    let host = host_c::host_results(&cc, &cases, &c_sources);

    let mut tally = Tally::default();
    for (i, c) in cases.iter().enumerate() {
        let got = gpu.run(&spirv[i].0, &spirv[i].1, &columns(c));
        assert_eq!(got.len(), host[i].len(), "{}: result length", c.name);
        for (j, (&g, &w)) in got.iter().zip(&host[i]).enumerate() {
            tally.cell(c, j, &[("device vs cpu-c", g, w)]);
        }
    }
    tally.finish("cpu_c_oracle", "unpopped-cpu-c", cases.len());
}

/// `slangc` compiles one emitted Slang kernel to SPIR-V words; `None` when `slangc` is absent
/// (a failure under `VULKANE_REQUIRE_SLANGC`).
fn slangc_words(
    dir: &std::path::Path,
    name: &str,
    source: &str,
    dt: ElementKind,
) -> Option<Vec<u32>> {
    let src = dir.join(format!("{name}.slang"));
    let spv = dir.join(format!("{name}.spv"));
    std::fs::write(&src, source).unwrap();
    let out = Command::new("slangc")
        .arg(&src)
        .args([
            "-entry", name, "-stage", "compute", "-target", "spirv", "-o",
        ])
        .arg(&spv)
        .args(vulkane_slang_emit::required_slangc_args(dt))
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_SLANGC").is_none(),
                "VULKANE_REQUIRE_SLANGC is set but `slangc` could not be run: {e}"
            );
            return None;
        }
    };
    assert!(
        out.status.success(),
        "slangc failed for {name}:\n{}\n--- source ---\n{source}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&spv).unwrap();
    assert_eq!(bytes.len() % 4, 0, "{name}: SPIR-V is whole words");
    Some(
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect(),
    )
}

/// Each case compiled through Slang. `Err` after a declared skip (no `slangc`); otherwise one entry
/// per case: the module, or `None` for a typed decline (collected in the second element).
#[allow(clippy::type_complexity)]
fn slang_modules(cases: &[Case]) -> Result<(Vec<Option<Vec<u32>>>, Vec<String>), ()> {
    let dir = tempfile::tempdir().unwrap();
    let (mut modules, mut declined) = (Vec::new(), Vec::new());
    for c in cases {
        match try_generate(&c.def, &c.key, &vulkane_slang_emit::Slang) {
            Ok(sl) => {
                let Some(words) = slangc_words(dir.path(), &sl.name, &sl.source, c.dt) else {
                    eprintln!(
                        "SKIP: `slangc` not on PATH (set VULKANE_REQUIRE_SLANGC to make this fatal)"
                    );
                    return Err(());
                };
                modules.push(Some(words));
            }
            Err(e) => {
                // A typed decline is a finding about the Slang emitter, not a disagreement.
                declined.push(format!("{}: {e:?}", c.name));
                modules.push(None);
            }
        }
    }
    Ok((modules, declined))
}

/// Run every case the Slang emitter spelled on the device through both emitters and tally the
/// Slang result against the SPIR-V result and against the host C result.
fn cross_check(
    gpu: &gpu::Gpu,
    cases: &[Case],
    spirv: &[Module],
    slang: &[Option<Vec<u32>>],
    host: &[Vec<u32>],
) -> Tally {
    let mut tally = Tally::default();
    for (i, c) in cases.iter().enumerate() {
        let Some(words) = &slang[i] else {
            continue;
        };
        let a = gpu.run(&spirv[i].0, &spirv[i].1, &columns(c));
        // `slangc` names every SPIR-V entry point `main`, whatever `-entry` selected.
        let b = gpu.run("main", words, &columns(c));
        for j in 0..a.len() {
            let checks = [
                ("slang vs spirv", b[j], a[j]),
                ("slang vs host", b[j], host[i][j]),
            ];
            tally.cell(c, j, &checks);
        }
    }
    tally
}

/// The cross-check: the same `OpDef` through the Slang emitter (compiled by `slangc`) and the
/// SPIR-V emitter, both run on the device. Each is held to the host C result, and the two are held
/// to each other, with the same per-op tolerance and domain as the oracle above.
#[test]
fn slang_and_spirv_emitters_agree_on_a_device() {
    common::require_serialization_lock();
    let Some(cc) = cc::find_compiler() else {
        eprintln!("SKIP: no host C compiler");
        return;
    };
    let Some(gpu) = gpu::Gpu::open("slang_spirv_cross") else {
        return;
    };
    let cases = all_cases();
    let (spirv, c_sources) = generate(&cases);
    let Ok((slang, declined)) = slang_modules(&cases) else {
        return;
    };
    let host = host_c::host_results(&cc, &cases, &c_sources);

    let tally = cross_check(&gpu, &cases, &spirv, &slang, &host);
    eprintln!(
        "slang_spirv_cross: the Slang emitter declined {} of {} cases (not compared):\n{}",
        declined.len(),
        cases.len(),
        declined.join("\n")
    );
    // Pinned: the Slang emitter spells every case here (U32 bitwise/shift ops were typed declines
    // until `slang_binary` routed U32 to the integer speller). A new decline is a regression.
    assert!(
        declined.is_empty(),
        "the Slang emitter declined cases the SPIR-V emitter spells:\n{}",
        declined.join("\n")
    );
    tally.finish(
        "slang_spirv_cross",
        "SPIR-V and host C",
        cases.len() - declined.len(),
    );
}
