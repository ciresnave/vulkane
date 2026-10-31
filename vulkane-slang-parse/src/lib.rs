// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Slang front end for [Unpopped](https://crates.io/crates/unpopped).
//!
//! Vulkane owns Slang in the CireSnave ecosystem, so the language's parser lives here
//! (the emitter is `vulkane-slang-emit`). Unpopped keeps the language-neutral half:
//! [`Frontend`] and the `lift_*` walkers in `unpopped::convert`, which this crate feeds with
//! a grammar binding, exactly as `baracuda-cuda-parse` does for CUDA.
//!
//! * [`parse_slang`] - Slang source to an error-tolerant tree-sitter CST.
//! * [`SLANG_RESIDUE`] - constructs refused as inexpressible (shared memory, barriers, atomics).
//! * [`SLANG`] - the [`Frontend`]: `numthreads` kernels with `output` / `input{K}` buffers.
//!   The buffer names are not a style choice: `out` and `in` are HLSL keywords.
//!
//! ```
//! use unpopped::convert::lift_elementwise;
//! use unpopped_vocab::ElementKind;
//! use vulkane_slang_parse::SLANG;
//!
//! let src = "StructuredBuffer<float> input0;\n\
//!     RWStructuredBuffer<float> output;\n\
//!     [numthreads(256, 1, 1)]\n\
//!     void copy(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; output[i] = input0[i]; }";
//! let lifted = lift_elementwise(&SLANG, src, "copy", &[ElementKind::F32]).unwrap();
//! assert_eq!(lifted.n_inputs, 1);
//! ```
//!
//! The grammar crate (`tree-sitter-slang`) is named in exactly one place, [`parse_slang`].
//! Moved here from `unpopped::convert` (Unpopped `213df4f`, v0.34.11); see
//! `THIRD-PARTY-NOTICES.md`.

use tree_sitter::{Parser, Tree};
use unpopped::convert::Frontend;

/// Parse Slang source into a tree-sitter CST.
///
/// `None` only if the grammar fails to load or the parser is cancelled; syntax errors
/// do not return `None`, they appear as `ERROR` nodes in the tree.
#[must_use]
pub fn parse_slang(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_slang::LANGUAGE_SLANG.into())
        .ok()?;
    parser.parse(src, None)
}

/// Slang constructs that aren't IR-expressible - group-shared memory, barriers, atomics.
pub const SLANG_RESIDUE: &[&str] = &[
    "groupshared",
    "GroupMemoryBarrier",
    "DeviceMemoryBarrier",
    "AllMemoryBarrier",
    "InterlockedAdd",
    "InterlockedCompareExchange",
    "RWByteAddressBuffer",
];

/// Slang: `numthreads` compute kernels, `output`/`input{K}` buffers.
///
/// The buffer names differ from CUDA's because `out` and `in` are **HLSL keywords** - a
/// language constraint, not a style choice.
pub const SLANG: Frontend = Frontend::new(
    "slang",
    "numthreads",
    SLANG_RESIDUE,
    "output",
    "input",
    parse_slang,
);

#[cfg(test)]
mod tests {
    use super::*;
    use unpopped::LiftError;
    use unpopped::convert::{lift, lift_elementwise, lift_reduction, lift_scan};
    use unpopped::ir::{Access, ReduceOp, ScalarExpr};
    use unpopped_vocab::ElementKind;

    const F32: &[ElementKind] = &[ElementKind::F32];

    const SLANG_MUL: &str = "StructuredBuffer<float> input0;\n\
        StructuredBuffer<float> input1;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256, 1, 1)]\n\
        void mul(uint3 tid : SV_DispatchThreadID) {\n\
            uint i = tid.x;\n\
            output[i] = input0[i] * input1[i];\n\
        }";
    const SLANG_COPY: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256, 1, 1)]\n\
        void k(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; output[i] = input0[i]; }";
    const SLANG_SUM: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256,1,1)]\n\
        void rsum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++) acc += input0[i]; output[0] = acc; }";
    const SLANG_CSUM: &str = "StructuredBuffer<float> input0;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256,1,1)]\n\
        void csum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++){ acc += input0[i]; output[i] = acc; } }";

    #[test]
    fn grammar_loads_and_parses() {
        let tree = parse_slang("float4 f(float x) { return x; }").expect("grammar loads");
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn syntax_errors_are_error_nodes_not_none() {
        let tree = parse_slang("void f( {").expect("a parse tree even for broken source");
        assert!(
            tree.root_node().has_error(),
            "broken source must show an ERROR node"
        );
    }

    #[test]
    fn slang_lifts_multiply() {
        let lifted = lift_elementwise(&SLANG, SLANG_MUL, "slang_mul", F32).unwrap();
        assert_eq!(lifted.n_inputs, 2);
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            )
        );
    }

    #[test]
    fn every_residue_marker_is_refused_as_inexpressible() {
        for marker in SLANG_RESIDUE {
            let src = format!(
                "[numthreads(256,1,1)]\nvoid k(uint3 tid : SV_DispatchThreadID) {{ {marker}; output[tid.x] = input0[tid.x]; }}"
            );
            assert!(
                matches!(
                    lift_elementwise(&SLANG, &src, "x", F32),
                    Err(LiftError::Inexpressible(ref w)) if w == marker
                ),
                "{marker}: {:?}",
                lift_elementwise(&SLANG, &src, "x", F32)
            );
        }
    }

    #[test]
    fn slang_lifts_sum_reduction() {
        let lifted = lift_reduction(&SLANG, SLANG_SUM, "rsum", F32).unwrap();
        assert_eq!(lifted.op.body, ScalarExpr::Input(0));
        assert!(matches!(
            lifted.op.access,
            Access::Reduction {
                op: ReduceOp::Sum,
                ..
            }
        ));
    }

    #[test]
    fn slang_lifts_cumsum() {
        let lifted = lift_scan(&SLANG, SLANG_CSUM, "csum", F32).unwrap();
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
        assert!(lift_elementwise(&SLANG, SLANG_SUM, "x", F32).is_err());
        assert!(lift_reduction(&SLANG, SLANG_COPY, "x", F32).is_err());
        assert!(lift_scan(&SLANG, SLANG_CSUM, "x", F32).is_ok());
        assert!(lift_reduction(&SLANG, SLANG_CSUM, "x", F32).is_err());
        assert!(lift_elementwise(&SLANG, SLANG_CSUM, "x", F32).is_err());
        assert!(lift(&SLANG, SLANG_SUM, "x", F32).is_ok());
        assert!(lift(&SLANG, SLANG_COPY, "x", F32).is_ok());
    }

    #[test]
    fn source_without_numthreads_is_not_a_kernel() {
        assert!(matches!(
            lift_elementwise(&SLANG, "float f(float x) { return x; }", "x", F32),
            Err(LiftError::NotAKernel)
        ));
    }
}
