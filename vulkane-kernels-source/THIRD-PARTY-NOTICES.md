# Third-party notices

Written for this crate when it moved from Fuel (`fuel-kernels-source`, Fuel `e3ff3427`). It replaces the
copy Fuel shipped, which was a byte-identical copy of Fuel's repository-wide notices (llama.cpp, candle,
bitsandbytes, Qwen weights) and named no file of this crate; that copy is in this crate's git history.
Licence texts are reproduced verbatim from upstream at the stated ref.

## Method

Every file in `kernels/` was checked for third-party origin three ways (measured 2026-10-30 against the
files at the import commit):

1. Fuel's own table (`docs/provenance/FILES.tsv` at `e3ff3427`): all 178 kernel-crate source files are
   classed `own` ("0 of N lines verbatim in candle"); the three packaged licence/notice files are `copy`.
2. A search of every file for origin words (llama, ggml, candle, copyright, derived, adapted, based on, URLs),
   which found 10 files with a hit; each is dispositioned below.
3. Normalised-line overlap (trimmed lines longer than 25 characters, comments and boilerplate excluded)
   of every `.slang` and `.glsl` file against llama.cpp's `ggml-quants.c` and `ggml-impl.h`, and against all of
   llama.cpp's `ggml-vulkan/vulkan-shaders/` (171 files), all at `9d57ce456c`: 0 verbatim lines in every file
   except `unary.slang`, whose single overlapping line is the sigmoid expression `return 1.0f / (1.0f + exp(-x));`.
   The control: the same procedure finds 50 of 50 lines when run on a slice of the reference itself.
   Limit: this finds copied lines, not translation between languages. The Slang kernels are a translation
   target, so a zero here does not clear a file, which is why step 2 and the dispositions below exist.

## llama.cpp / ggml

- Origin: https://github.com/ggml-org/llama.cpp, `LICENSE` at commit `9d57ce456c94d241dde672b2db9cf18879766568`.
- **Derived:** `kernels/dequant_q4_km.slang`. `get_scale_min_k4` is a line-by-line Slang translation of llama.cpp's
  function of the same name (`ggml/src/ggml-quants.c`, line 880 at that commit: same branch on `j < 4`, same masks and
  shifts), and the decode order follows `dequantize_row_q4_K`. An origin comment is in the file.
- **Format only, not code:** `kernels/dequant_q4_0.slang`, `dequant_q8_0.slang`, `quantize_q8_0.slang` implement
  the GGML Q4_0 / Q8_0 block layouts, which are a data format. They name GGML in their comments; 0 verbatim lines.
  Listed so a reader sees they were checked, not because code was found.

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## webgpu-blas (acknowledgement; no code taken)

- Origin: https://github.com/milhidaka/webgpu-blas at commit `af28338346fc66766f4bbe37b01d25154eb14f51`
  (HEAD when checked), MIT, Copyright (c) 2019 Masatoshi Hidaka.
- `kernels/matmul.slang` says it is "based on webgpu-blas (MIT) optimization pattern". Compared with the
  upstream shaders `src/shader_sgemm_block.ts` and `src/shader_sgemm_generic.ts`: 0 of 31 lines (over 25
  characters) appear in the upstream (control: the same method finds 74 of 74 when a file is compared with
  itself). The structures differ (ours: scalar 4x4 register tile, 16x16 workgroup, strided batch; upstream:
  `vec4` tiles, 8x8 workgroup). Only the register-tiling idea is shared, and an idea is not licensed text, but the
  author is credited here because the file's own comment claims the debt. Upstream's licence follows, verbatim.

```text
MIT License

Copyright (c) 2019 Masatoshi Hidaka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Everything else: first-party

The remaining `.slang` and `.glsl` files, `compile.sh` and `hoist_extinst_imports.py` are Fuel-authored
(68 commits by CireSnave, 6 by "Eric Evans' AI Agent", 4 by Eric Evans in the imported history) and carry
`SPDX-License-Identifier: MIT OR Apache-2.0`. Fuel's table found no lineage to candle for any of them.
This repository's `LICENSE-MIT` and `LICENSE-APACHE` are shipped in the package. Fuel's `LICENSE-MIT` (candle's
text, with no copyright holder) is in the git history, not shipped.
