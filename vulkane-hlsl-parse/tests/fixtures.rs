// SPDX-License-Identifier: MIT OR Apache-2.0
//! The hand-written pure-HLSL fixtures (design note H2): each is parsed and its ERROR / MISSING
//! nodes are counted. The one known grammar gap (`cbuffer N : register(bK)` gives `MISSING ;`)
//! is pinned, so a grammar release that fixes it fails this test and the residue note is retired.

use tree_sitter::Node;
use vulkane_hlsl_parse::parse_hlsl;

fn bad_nodes(n: Node<'_>) -> usize {
    let own = usize::from(n.is_error() || n.is_missing());
    let mut cur = n.walk();
    own + n.children(&mut cur).map(bad_nodes).sum::<usize>()
}

fn count(src: &str) -> usize {
    bad_nodes(parse_hlsl(src).expect("grammar loads").root_node())
}

#[test]
fn fixtures_without_the_known_gap_parse_clean() {
    for (name, src) in [
        ("mul", include_str!("fixtures/mul.hlsl")),
        ("rsum", include_str!("fixtures/rsum.hlsl")),
        ("csum", include_str!("fixtures/csum.hlsl")),
        (
            "semantics_struct",
            include_str!("fixtures/semantics_struct.hlsl"),
        ),
    ] {
        assert_eq!(count(src), 0, "{name} must parse with 0 ERROR/MISSING");
    }
}

#[test]
fn the_cbuffer_register_gap_is_pinned() {
    let src = include_str!("fixtures/cbuffer_groupshared.hlsl");
    assert!(
        count(src) >= 1,
        "gap fixed upstream? retire the residue note and tighten this test"
    );
    // Control: the same cbuffer without `: register(..)` is clean.
    assert_eq!(count("cbuffer Params { uint n; };"), 0);
}

#[test]
fn the_tbuffer_register_gap_is_pinned() {
    // Measured by this crate (2026-11-05): `tbuffer N : register(tK)` has the same `MISSING ;` gap.
    assert!(count("tbuffer Table : register(t2) { float4 w[4]; };") >= 1);
    assert_eq!(count("tbuffer Table { float4 w[4]; };"), 0);
}

#[test]
fn control_broken_source_is_counted() {
    assert!(count("int main(") >= 1);
}
