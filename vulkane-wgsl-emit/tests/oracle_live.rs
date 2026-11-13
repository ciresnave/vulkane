// SPDX-License-Identifier: MIT OR Apache-2.0
//! The behaviour oracle (`docs/design/wgsl-parser-emitter.md`, W4): the same `OpDef` is lowered by
//! `unpopped-cpu-c` (portable C, compiled and run on the host) and by this emitter (WGSL, compiled
//! by `naga` to SPIR-V and run on a live device) over the same input bits, and the two result
//! bit-streams are compared cell by cell. Cases, tolerances and domains are the SPIR-V emitter's
//! (`vulkane-spirv-emit/tests/support`), so the two device oracles hold the same bar.
//!
//! WGSL has no f64. Without a device or a host C compiler the test follows `VULKANE_REQUIRE_DEVICE` and
//! `CI` rather than passing silently. One WGSL compiler only (naga).

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

use unpopped::try_generate;
use unpopped_cpu_c::CpuC;
use vulkane_wgsl_emit::Wgsl;

use cases::{Case, agree, cases as all_cases};

/// Compile one emitted WGSL kernel to SPIR-V words with `naga` (the only WGSL compiler available here; Tint is not
/// installed). Parse and validation failures are test failures, not skips.
fn naga_words(name: &str, source: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
        panic!(
            "naga rejects {name}:
{}
{source}",
            e.emit_to_string(source)
        )
    });
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| {
        panic!(
            "naga validation of {name}: {e:?}
{source}"
        )
    });
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .unwrap_or_else(|e| panic!("naga SPIR-V output for {name}: {e:?}"))
}

/// A kernel ready to run on the device: its entry-point name and SPIR-V words.
type Module = (String, Vec<u32>);

/// Each case's `naga`-compiled WGSL module and its `unpopped-cpu-c` kernel `(name, C source)`.
fn generate(cases: &[Case]) -> (Vec<Module>, Vec<(String, String)>) {
    let (mut modules, mut c_sources) = (Vec::new(), Vec::new());
    for c in cases {
        let h = try_generate(&c.def, &c.key, &Wgsl)
            .unwrap_or_else(|e| panic!("{}: the WGSL backend declined: {e:?}", c.name));
        modules.push((h.name.clone(), naga_words(&h.name, &h.source)));
        let k = try_generate(&c.def, &c.key, &CpuC)
            .unwrap_or_else(|e| panic!("{}: the CPU-C backend declined: {e:?}", c.name));
        c_sources.push((k.name.clone(), k.source.clone()));
    }
    (modules, c_sources)
}

/// WGSL defines `a % b` as `a - b * trunc(a / b)` and bounds `/` by 2.5 ULP, so a quotient within a few ULP of an integer may
/// land on either side of it and the remainder jumps by a whole divisor (measured: `1.5 % 1.5` is `1.5`, not `0`, on the test
/// device when the module does not ask for IEEE division). The floored `Rem` has the same dependence. Those rows are not
/// specified by WGSL, so a disagreement there is reported, not failed. Every other row of both ops is compared as before.
fn remainder_near_a_quotient_tie(c: &Case, row: &[u32]) -> bool {
    if !c.name.starts_with("b_rem") {
        return false;
    }
    let q = f32::from_bits(row[0]) / f32::from_bits(row[1]);
    q.is_finite() && (q - q.round()).abs() <= q.abs() * 5e-7
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
        let in_domain = c.domain.holds(&row) && !remainder_near_a_quotient_tie(c, &row);
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
            "wgsl_cpu_c_oracle: {} out-of-domain cells disagree (reported, not failed)",
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
            "wgsl_cpu_c_oracle: {} in-domain cells over {kernels} kernels agree",
            self.compared
        );
    }
}

#[test]
fn emitted_wgsl_matches_the_cpu_c_oracle_on_a_device() {
    common::require_serialization_lock();
    let Some(cc) = cc::find_compiler() else {
        eprintln!("SKIP: no host C compiler");
        return;
    };
    let Some(gpu) = gpu::Gpu::open("wgsl_cpu_c_oracle") else {
        return; // skip already declared
    };
    let cases = all_cases();
    assert!(
        !cases.is_empty(),
        "the shared case list is empty: nothing would be compared"
    );
    let (modules, c_sources) = generate(&cases);
    let host = host_c::host_results(&cc, &cases, &c_sources);

    let mut tally = Tally::default();
    for (i, c) in cases.iter().enumerate() {
        let cols: Vec<&[u32]> = c.inputs.iter().map(Vec::as_slice).collect();
        // naga names the SPIR-V entry point after the WGSL function.
        let got = gpu.run(&modules[i].0, &modules[i].1, &cols);
        assert_eq!(got.len(), host[i].len(), "{}: result length", c.name);
        for (j, (&g, &w)) in got.iter().zip(&host[i]).enumerate() {
            tally.cell(c, j, g, w);
        }
    }
    tally.finish(cases.len());
}
