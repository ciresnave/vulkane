// SPDX-License-Identifier: MIT OR Apache-2.0
//! The HLSL front end for [Unpopped](https://crates.io/crates/unpopped).
//!
//! Vulkane owns HLSL in the CireSnave ecosystem (ruling 2026-10-30), so the language's parser lives
//! here; the emitter will be `vulkane-hlsl-emit`. Unpopped keeps the language-neutral half:
//! [`Frontend`] and the `lift_*` walkers in `unpopped::convert`, which this crate feeds with a
//! grammar binding, exactly as `fishstringer-slang-parse` (formerly `vulkane-slang-parse`) does for Slang.
//!
//! * [`parse_hlsl`] - HLSL source to an error-tolerant tree-sitter CST.
//! * [`HLSL_RESIDUE`] - constructs refused as inexpressible (shared memory, barriers, atomics).
//! * [`HLSL`] - the [`Frontend`]: `numthreads` kernels with `output` / `input{K}` buffers.
//!   The buffer names are not a style choice: `out` and `in` are HLSL keywords.
//!
//! ```
//! use unpopped::convert::lift_elementwise;
//! use unpopped_vocab::ElementKind;
//! use vulkane_hlsl_parse::HLSL;
//!
//! let src = "StructuredBuffer<float> input0;\n\
//!     RWStructuredBuffer<float> output;\n\
//!     [numthreads(256, 1, 1)]\n\
//!     void copy(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; output[i] = input0[i]; }";
//! let lifted = lift_elementwise(&HLSL, src, "copy", &[ElementKind::F32]).unwrap();
//! assert_eq!(lifted.n_inputs, 1);
//! ```
//!
//! The grammar crate (`tree-sitter-hlsl`) is named in exactly one place, [`parse_hlsl`].
//!
//! Known grammar residue (measured 2026-11-05, `docs/design/hlsl-parser-emitter.md` H1): the grammar
//! reports `MISSING ;` for `cbuffer N : register(bK) { .. }` and an `ERROR` for the attribute
//! `[shader("compute")]`. Both are tolerated, not patched: the lifters walk the tree for the kernel
//! function and do not require a clean parse.

use tree_sitter::{Parser, Tree};
use unpopped::convert::Frontend;

/// Parse HLSL source into a tree-sitter CST.
///
/// `None` only if the grammar fails to load or the parser is cancelled; syntax errors
/// do not return `None`, they appear as `ERROR` nodes in the tree.
#[must_use]
pub fn parse_hlsl(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_hlsl::LANGUAGE_HLSL.into())
        .ok()?;
    parser.parse(src, None)
}

/// HLSL constructs that aren't IR-expressible - group-shared memory, barriers, atomics.
pub const HLSL_RESIDUE: &[&str] = &[
    "groupshared",
    "GroupMemoryBarrier",
    "DeviceMemoryBarrier",
    "AllMemoryBarrier",
    "InterlockedAdd",
    "InterlockedCompareExchange",
    "RWByteAddressBuffer",
];

/// HLSL: `numthreads` compute kernels, `output`/`input{K}` buffers.
///
/// The buffer names differ from CUDA's because `out` and `in` are **HLSL keywords** - a
/// language constraint, not a style choice.
pub const HLSL: Frontend = Frontend::new(
    "hlsl",
    "numthreads",
    HLSL_RESIDUE,
    "output",
    "input",
    parse_hlsl,
);

#[cfg(test)]
mod tests {
    use super::*;
    use unpopped::LiftError;
    use unpopped::convert::{lift_elementwise, lift_reduction, lift_scan};
    use unpopped::ir::{Access, ReduceOp, ScalarExpr};
    use unpopped_vocab::ElementKind;

    const F32: &[ElementKind] = &[ElementKind::F32];

    #[test]
    fn grammar_loads_and_parses() {
        let tree = parse_hlsl("float4 f(float x) { return x; }").expect("grammar loads");
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn syntax_errors_are_error_nodes_not_none() {
        let tree = parse_hlsl("int main(").expect("a parse tree even for broken source");
        assert!(
            tree.root_node().has_error(),
            "broken source must show an ERROR node"
        );
    }

    #[test]
    fn every_residue_marker_is_refused_as_inexpressible() {
        for marker in HLSL_RESIDUE {
            let src = format!(
                "[numthreads(256,1,1)]\nvoid k(uint3 tid : SV_DispatchThreadID) {{ {marker}; output[tid.x] = input0[tid.x]; }}"
            );
            assert!(
                matches!(
                    lift_elementwise(&HLSL, &src, "x", F32),
                    Err(LiftError::Inexpressible(ref w)) if w == marker
                ),
                "{marker}: {:?}",
                lift_elementwise(&HLSL, &src, "x", F32)
            );
        }
    }

    #[test]
    fn source_without_numthreads_is_not_a_kernel() {
        assert!(matches!(
            lift_elementwise(&HLSL, "float f(float x) { return x; }", "x", F32),
            Err(LiftError::NotAKernel)
        ));
    }

    #[test]
    fn lifts_multiply_sum_and_cumsum() {
        let mul = include_str!("../tests/fixtures/mul.hlsl");
        let lifted = lift_elementwise(&HLSL, mul, "mul", F32).unwrap();
        assert_eq!(lifted.n_inputs, 2);
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1))
            )
        );
        let sum = include_str!("../tests/fixtures/rsum.hlsl");
        let lifted = lift_reduction(&HLSL, sum, "rsum", F32).unwrap();
        assert!(matches!(
            lifted.op.access,
            Access::Reduction {
                op: ReduceOp::Sum,
                ..
            }
        ));
        let cs = include_str!("../tests/fixtures/csum.hlsl");
        let lifted = lift_scan(&HLSL, cs, "csum", F32).unwrap();
        assert!(matches!(
            lifted.op.access,
            Access::Scan {
                op: ReduceOp::Sum,
                ..
            }
        ));
    }

    #[test]
    fn the_three_lifters_do_not_cross() {
        let mul = include_str!("../tests/fixtures/mul.hlsl");
        let sum = include_str!("../tests/fixtures/rsum.hlsl");
        let cs = include_str!("../tests/fixtures/csum.hlsl");
        assert!(lift_elementwise(&HLSL, sum, "x", F32).is_err());
        assert!(lift_reduction(&HLSL, mul, "x", F32).is_err());
        assert!(lift_scan(&HLSL, cs, "x", F32).is_ok());
        assert!(lift_reduction(&HLSL, cs, "x", F32).is_err());
        assert!(lift_elementwise(&HLSL, cs, "x", F32).is_err());
    }
}
