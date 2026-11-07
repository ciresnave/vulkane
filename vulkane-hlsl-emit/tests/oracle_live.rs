// SPDX-License-Identifier: MIT OR Apache-2.0
//! H4, the behaviour oracle (`docs/design/hlsl-parser-emitter.md`): the same `OpDef` is lowered by
//! `unpopped-cpu-c` (portable C, compiled and run on the host) and by this emitter (HLSL, compiled
//! by `dxc` to SPIR-V and run on a live device) over the same input bits, and the two result
//! bit-streams are compared cell by cell. Cases, tolerances and domains are the SPIR-V emitter's
//! (`vulkane-spirv-emit/tests/support`), so the two device oracles hold the same bar.
//!
//! `f64` is not run (the shared test device does not request `shaderFloat64`). Without a device,
//! `dxc` or a host C compiler the test follows `VULKANE_REQUIRE_DEVICE`, `VULKANE_REQUIRE_DXC` and
//! `CI` rather than passing silently.

#[path = "../../vulkane-spirv-emit/tests/support/cases.rs"]
mod cases;
#[path = "../../vulkane-spirv-emit/tests/support/cc.rs"]
mod cc;
#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "../../vulkane-spirv-emit/tests/support/gpu.rs"]
mod gpu;
#[path = "../../vulkane-spirv-emit/tests/support/host_c.rs"]
mod host_c;

use std::path::Path;
use std::process::Command;

use unpopped::try_generate;
use unpopped_cpu_c::CpuC;
use vulkane_hlsl_emit::{Hlsl, required_dxc_args};

use cases::{Case, agree, cases as all_cases};

/// Compile one emitted HLSL kernel with `dxc`; `None` when `dxc` is absent (a failure under
/// `VULKANE_REQUIRE_DXC`).
fn dxc_words(dir: &Path, c: &Case, name: &str, source: &str) -> Option<Vec<u32>> {
    let src = dir.join(format!("{name}.hlsl"));
    let spv = dir.join(format!("{name}.spv"));
    std::fs::write(&src, source).unwrap();
    let out = Command::new("dxc")
        .args([
            "-T",
            "cs_6_2",
            "-E",
            name,
            "-spirv",
            "-fspv-target-env=vulkan1.1",
        ])
        .args(required_dxc_args(c.dt))
        .arg(&src)
        .arg("-Fo")
        .arg(&spv)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_DXC").is_none(),
                "VULKANE_REQUIRE_DXC is set but `dxc` could not be run: {e}"
            );
            return None;
        }
    };
    assert!(
        out.status.success(),
        "dxc failed for {name}:\n{}\n--- source ---\n{source}",
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

/// A kernel ready to run on the device: its entry-point name and SPIR-V words.
type Module = (String, Vec<u32>);

/// Each case's `dxc`-compiled HLSL module and its `unpopped-cpu-c` kernel `(name, C source)`.
/// `None` after a declared skip (no `dxc`).
fn generate(cases: &[Case]) -> Option<(Vec<Module>, Vec<(String, String)>)> {
    let dir = tempfile::tempdir().unwrap();
    let (mut modules, mut c_sources) = (Vec::new(), Vec::new());
    for c in cases {
        let h = try_generate(&c.def, &c.key, &Hlsl)
            .unwrap_or_else(|e| panic!("{}: the HLSL backend declined: {e:?}", c.name));
        let Some(words) = dxc_words(dir.path(), c, &h.name, &h.source) else {
            eprintln!("SKIP: `dxc` not on PATH (set VULKANE_REQUIRE_DXC to make this fatal)");
            return None;
        };
        modules.push((h.name.clone(), words));
        let k = try_generate(&c.def, &c.key, &CpuC)
            .unwrap_or_else(|e| panic!("{}: the CPU-C backend declined: {e:?}", c.name));
        c_sources.push((k.name.clone(), k.source.clone()));
    }
    Some((modules, c_sources))
}

/// Disagreements found so far, split by whether the op is specified on that input row.
#[derive(Default)]
struct Tally {
    /// In-domain cells compared.
    compared: usize,
    /// In-domain disagreements: these fail the test.
    bad: Vec<String>,
    /// Out-of-domain disagreements: reported only.
    undefined: usize,
}

impl Tally {
    /// Check device cell `got` against host cell `want` on row `j` of `c`.
    fn cell(&mut self, c: &Case, j: usize, got: u32, want: u32) {
        let row: Vec<u32> = c.inputs.iter().map(|col| col[j]).collect();
        let in_domain = c.domain.holds(&row);
        self.compared += usize::from(in_domain);
        if agree(c.tol, c.dt, got, want) {
            return;
        }
        if in_domain {
            self.bad.push(format!(
                "{} inputs {row:x?}: {got:#010x} vs {want:#010x}",
                c.name
            ));
        } else {
            self.undefined += 1;
        }
    }

    /// Fail on any in-domain disagreement, and on a run that compared nothing (Pip, PR #146).
    fn finish(&self, kernels: usize) {
        assert!(self.compared > 0, "no in-domain cell was compared");
        eprintln!(
            "hlsl_cpu_c_oracle: {} out-of-domain cells disagree (reported, not failed)",
            self.undefined
        );
        assert!(
            self.bad.is_empty(),
            "{} of {} in-domain cells disagree with unpopped-cpu-c:
{}",
            self.bad.len(),
            self.compared,
            self.bad.join(
                "
"
            )
        );
        eprintln!(
            "hlsl_cpu_c_oracle: {} in-domain cells over {kernels} kernels agree",
            self.compared
        );
    }
}

#[test]
fn emitted_hlsl_matches_the_cpu_c_oracle_on_a_device() {
    common::require_serialization_lock();
    let Some(cc) = cc::find_compiler() else {
        eprintln!("SKIP: no host C compiler");
        return;
    };
    let Some(gpu) = gpu::Gpu::open("hlsl_cpu_c_oracle") else {
        return; // skip already declared
    };
    let cases = all_cases();
    assert!(
        !cases.is_empty(),
        "the shared case list is empty: nothing would be compared"
    );
    let Some((modules, c_sources)) = generate(&cases) else {
        return;
    };
    let host = host_c::host_results(&cc, &cases, &c_sources);

    let mut tally = Tally::default();
    for (i, c) in cases.iter().enumerate() {
        let cols: Vec<&[u32]> = c.inputs.iter().map(Vec::as_slice).collect();
        // `dxc -E <name>` keeps the entry-point name, unlike `slangc`.
        let got = gpu.run(&modules[i].0, &modules[i].1, &cols);
        assert_eq!(got.len(), host[i].len(), "{}: result length", c.name);
        for (j, (&g, &w)) in got.iter().zip(&host[i]).enumerate() {
            tally.cell(c, j, g, w);
        }
    }
    tally.finish(cases.len());
}
