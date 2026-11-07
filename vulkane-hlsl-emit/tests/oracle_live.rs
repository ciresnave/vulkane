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
    let dir = tempfile::tempdir().unwrap();
    let (mut modules, mut c_sources) = (Vec::new(), Vec::new());
    for c in &cases {
        let h = try_generate(&c.def, &c.key, &Hlsl)
            .unwrap_or_else(|e| panic!("{}: the HLSL backend declined: {e:?}", c.name));
        let Some(words) = dxc_words(dir.path(), c, &h.name, &h.source) else {
            eprintln!("SKIP: `dxc` not on PATH (set VULKANE_REQUIRE_DXC to make this fatal)");
            return;
        };
        modules.push((h.name.clone(), words));
        let k = try_generate(&c.def, &c.key, &CpuC)
            .unwrap_or_else(|e| panic!("{}: the CPU-C backend declined: {e:?}", c.name));
        c_sources.push((k.name.clone(), k.source.clone()));
    }
    let host = host_c::host_results(&cc, &cases, &c_sources);

    let (mut compared, mut bad, mut undefined) = (0usize, Vec::new(), 0usize);
    for (i, c) in cases.iter().enumerate() {
        let cols: Vec<&[u32]> = c.inputs.iter().map(Vec::as_slice).collect();
        // `dxc -E <name>` keeps the entry-point name, unlike `slangc`.
        let got = gpu.run(&modules[i].0, &modules[i].1, &cols);
        assert_eq!(got.len(), host[i].len(), "{}: result length", c.name);
        for (j, (&g, &w)) in got.iter().zip(&host[i]).enumerate() {
            let row: Vec<u32> = c.inputs.iter().map(|col| col[j]).collect();
            let in_domain = c.domain.holds(&row);
            compared += usize::from(in_domain);
            if agree(c.tol, c.dt, g, w) {
                continue;
            }
            if in_domain {
                bad.push(format!(
                    "{} inputs {row:x?}: {g:#010x} vs {w:#010x}",
                    c.name
                ));
            } else {
                undefined += 1;
            }
        }
    }
    // A run that compared nothing is not a pass (Pip, PR #146).
    assert!(compared > 0, "no in-domain cell was compared");
    eprintln!("hlsl_cpu_c_oracle: {undefined} out-of-domain cells disagree (reported, not failed)");
    assert!(
        bad.is_empty(),
        "{} of {compared} in-domain cells disagree with unpopped-cpu-c:\n{}",
        bad.len(),
        bad.join("\n")
    );
    eprintln!(
        "hlsl_cpu_c_oracle: {compared} in-domain cells over {} kernels agree",
        cases.len()
    );
}
