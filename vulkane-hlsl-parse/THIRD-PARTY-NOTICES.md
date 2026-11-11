# Third-party notices

**No third-party source is copied into this crate.**

## Origin

- `parse_hlsl`, `HLSL_RESIDUE` and the `HLSL` frontend are modelled on this repository's former `vulkane-slang-parse` (now FishStringer's `fishstringer-slang-parse`)
  (same owner, `MIT OR Apache-2.0`), which in turn re-created them from Unpopped's `convert.rs`. The HLSL residue
  list is the same seven strings as the Slang one, because Slang's kernel subset is HLSL's.
- The five `tests/fixtures/*.hlsl` files were written for this crate (2026-11-05) and each compiles with
  `dxc -T cs_6_2 -spirv -fspv-target-env=vulkan1.1` (dxc 1.9.0.5347).

## Dependencies (not copied or redistributed here)

- `tree-sitter-hlsl` 0.2 (MIT, https://github.com/theHamsta/tree-sitter-hlsl).
- `tree-sitter` (MIT) and `unpopped` (`MIT OR Apache-2.0`).

Every source file declares `SPDX-License-Identifier: MIT OR Apache-2.0`.
