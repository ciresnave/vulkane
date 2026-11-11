// SPDX-License-Identifier: MIT OR Apache-2.0
//! The compute corpus: every fixture must parse here, validate in naga (the oracle), and reflect to
//! the same entry points and bindings naga sees.

use vulkane_wgsl_parse::{Dim, Stage, parse_wgsl};

const FIXTURES: [(&str, &str); 3] = [
    ("c_elementwise", include_str!("fixtures/c_elementwise.wgsl")),
    ("c_reduction", include_str!("fixtures/c_reduction.wgsl")),
    ("c_scan", include_str!("fixtures/c_scan.wgsl")),
];

fn naga_module(src: &str) -> naga::Module {
    let m = naga::front::wgsl::parse_str(src).expect("naga parses the fixture");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&m)
    .expect("naga validates the fixture");
    m
}

#[test]
fn every_fixture_reflects_what_naga_sees() {
    for (name, src) in FIXTURES {
        let ours = parse_wgsl(src).unwrap_or_else(|e| panic!("{name}: {e}"));
        let theirs = naga_module(src);
        let mut want: Vec<(String, [u32; 3])> = theirs
            .entry_points
            .iter()
            .map(|e| (e.name.clone(), e.workgroup_size))
            .collect();
        let mut got: Vec<(String, [u32; 3])> = ours
            .entry_points
            .iter()
            .map(|e| {
                assert_eq!(e.stage, Stage::Compute, "{name}");
                let w = e.workgroup_size.clone().expect("compute has a size");
                let d = |x: &Option<Dim>| match x {
                    Some(Dim::Literal(v)) => *v,
                    None => 1,
                    other => panic!("{name}: non-literal {other:?}"),
                };
                (e.name.clone(), [d(&w[0]), d(&w[1]), d(&w[2])])
            })
            .collect();
        want.sort();
        got.sort();
        assert!(
            !want.is_empty(),
            "{name}: control: naga sees an entry point"
        );
        assert_eq!(got, want, "{name}: entry points");

        let mut want_b: Vec<(u32, u32)> = theirs
            .global_variables
            .iter()
            .filter_map(|(_, g)| g.binding.map(|b| (b.group, b.binding)))
            .collect();
        let mut got_b: Vec<(u32, u32)> =
            ours.bindings.iter().map(|b| (b.group, b.binding)).collect();
        want_b.sort();
        got_b.sort();
        assert_eq!(got_b, want_b, "{name}: bindings");
    }
}

#[test]
fn reduction_reflects_known_shape() {
    let m = parse_wgsl(FIXTURES[1].1).unwrap();
    assert_eq!(m.entry_points[0].name, "sum_f32");
    assert_eq!(
        m.entry_points[0].workgroup_size,
        Some([
            Some(Dim::Literal(256)),
            Some(Dim::Literal(1)),
            Some(Dim::Literal(1))
        ])
    );
    let b = &m.bindings;
    assert_eq!(b.len(), 3);
    assert_eq!(
        (b[2].name.as_str(), b[2].group, b[2].binding),
        ("params", 0, 2)
    );
    assert_eq!(b[0].ty, "array<f32>");
}

#[test]
fn syntax_error_is_an_error_not_an_empty_module() {
    let e = parse_wgsl("@compute fn (").unwrap_err();
    assert!(!e.message.is_empty());
}

#[test]
fn non_literal_binding_is_refused() {
    let e =
        parse_wgsl("const G = 0u;\n@group(G) @binding(0) var<storage> a: array<f32>;").unwrap_err();
    assert!(e.message.contains("@group"), "{e}");
}

#[test]
fn override_workgroup_size_is_kept_as_text() {
    let m = parse_wgsl("override N: u32 = 64;\n@compute @workgroup_size(N) fn f() {}").unwrap();
    assert_eq!(
        m.entry_points[0].workgroup_size.as_ref().unwrap()[0],
        Some(Dim::Expr("N".into()))
    );
}
