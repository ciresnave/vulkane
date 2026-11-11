# Third-party notices

Written for this crate when it moved from Fuel (`fuel-kernels-source`, Fuel `e3ff3427`); rewritten when the Slang
files left it. This crate now holds the 27 `.glsl` kernels and `kernels/compile.sh`. Licence texts are reproduced
verbatim from upstream where a notice applies; none applies to the files that remain (below).

## Slang files moved to FishStringer

CireSnave, 2026-11-06 (via the PM): FishStringer owns Slang. The 147 `.slang` files, the Slang emitter and parser, and
`hoist_extinst_imports.py` moved to `fishstringer-kernels-source`, `fishstringer-slang-emit` and
`fishstringer-slang-parse`, with their history. **Every third-party disposition that applied to those files
(llama.cpp / ggml for `dequant_q4_km.slang`, the format-only Q4_0 / Q8_0 files, the webgpu-blas
acknowledgement for `matmul.slang`, the candle check) went with them**, and so did the licence texts: they are in
`fishstringer-kernels-source`'s `THIRD-PARTY-NOTICES.md` (checked in the 0.0.13 package on 2026-11-07: it carries
both the ggml and the Hidaka licence texts). The version of this file that carries them here is in this repository's
git history (before the "Slang switch, step 2" change).

## What remains, and what was checked

Measured 2026-11-07 on `origin/main` `eb33173` plus this change:

1. An origin-word search (llama, ggml, candle, copyright, derived, adapted, based on, URLs) over every `.glsl` file
   and `compile.sh`: **0 files with a hit**. Control: the same search over the 147 `.slang` files at the same
   commit finds 10, so the pattern can match.
2. The earlier overlap measurement (2026-10-30, at the import commit) already covered these `.glsl` files: trimmed
   lines longer than 25 characters, comments and boilerplate excluded, against llama.cpp's `ggml-quants.c`,
   `ggml-impl.h` and `ggml-vulkan/vulkan-shaders/` at `9d57ce456c`, and against candle's `candle-kernels/` and
   `candle-metal-kernels/` at `549eacb011`: 0 verbatim lines in every `.glsl` file (control: 50 of 50 and 40 of 40
   when a reference is compared with itself). Limit: this finds copied lines, not translation between languages.
   It was not re-run after the move; the files are unchanged since.
3. Fuel's own table (`docs/provenance/FILES.tsv` at `e3ff3427`) classed every source file of the original crate
   `own` ("0 of N lines verbatim in candle").

## Everything here is first-party

The `.glsl` files and `compile.sh` are Fuel-authored and carry `SPDX-License-Identifier: MIT OR Apache-2.0`.
This repository's `LICENSE-MIT` and `LICENSE-APACHE` are shipped in the package. Fuel's `LICENSE-MIT` (candle's
text, with no copyright holder) is in the git history, not shipped.
