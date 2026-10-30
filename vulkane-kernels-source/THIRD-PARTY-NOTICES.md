# Third-party notices

Licence texts for third-party work this repository derives from, reproduced
verbatim from the upstream source at the stated ref. A file that derives from
one of these carries no SPDX identifier of ours; it is registered in
`fuel-ir/tests/spdx_header_ratchet.rs` (`INHERITED_WITH_NOTICE`), which fails
if the notice is removed.

## llama.cpp / ggml

- Origin: https://github.com/ggml-org/llama.cpp, `LICENSE` at commit `9d57ce456c`
  (the commit `GgmlDType` in `fuel-ir/src/quantized.rs` is pinned to).
- Also reached through candle (next section): the Q4_0 / Q8_0 / k-quant block layouts and quantization
  algorithms in `fuel-quantized/src/k_quants.rs`, `utils.rs`, `avx.rs`, `neon.rs`. Compared line by line only for
  `BlockQ4_0::from_float` (the same steps as `quantize_row_q4_0_ref` in `ggml/src/ggml-quants.c` at
  `9d57ce456c`); the rest rests on candle's own citations of llama.cpp commits in those files' comments.
- Used by: `fuel-quantized/src/iq4.rs` (IQ4_NL and IQ4_XS block layouts, codebook,
  dequantization arithmetic), `fuel-quantized/src/iq_grid_types.rs` (IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M block layouts and dequantization),
  `fuel-quantized/src/tq_mx_types.rs` (TQ1_0, TQ2_0, MXFP4, NVFP4 block layouts, codebook
  and scale decoding) and `fuel-quantized/src/iq_grids.rs` (the IQ lookup tables, generated
  from `ggml/src/ggml-common.h` by `scripts/gen_iq_grids.py`); all verified against
  `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`, `ggml/src/ggml-impl.h`,
  `ggml/include/ggml.h` at that commit, and the IQ dequantizers additionally byte-compared with llama.cpp's own
  compiled C (see GAP-211 in `docs/gaps.md`).

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

## candle (Hugging Face)

- Origin: https://github.com/huggingface/candle, dual-licensed MIT OR Apache-2.0. Fuel is a fork of it (this
  repository's history still carries candle's commits). Its `LICENSE-MIT` is byte-identical to this repository's
  root `LICENSE-MIT`, and its `LICENSE-APACHE` is the root `LICENSE-APACHE` apart from one leading blank line
  (both compared at upstream commit `549eacb011f25b2c6fc765d6c3dbd26e42f72ee6`, 2025-11-18, the upstream
  counterpart of `7b1d2a0a8`, the last candle-era commit here that touched these files). Neither upstream licence file names a holder.
- Used by (traced to `candle-core/src/quantized/` at that commit, then modified here):
  `fuel-quantized/src/k_quants.rs` (from `k_quants.rs`), `utils.rs` (from `utils.rs`), `avx.rs` (from `avx.rs`),
  `neon.rs` (from `neon.rs`), and, in part only, `cpu.rs` (the `QuantizedType` trait, from `mod.rs`).
  These files keep our SPDX line because the licence is the same one candle declares.
- Verified: the repository's own history blobs for those four files at `7b1d2a0a8` are byte-identical to
  upstream at `549eacb01`. Non-blank lines of the current file that also appear verbatim in the upstream file:
  `k_quants.rs` 2162 of 2257, `utils.rs` 428 of 518, `avx.rs` 613 of 752, `neon.rs` 536 of 697 (measured
  2026-10-25; `cpu.rs` 28 of 151 code lines over 25 characters against all of upstream `quantized/*.rs`).
- Not derived from candle by the same measure: `dequant.rs` (0 of 94 such lines verbatim; it dispatches to
  `GgmlType::to_float` and holds no block math), `awq_repack.rs` (written against AutoAWQ's layout and
  baracuda's kernel layout), `lib.rs`. Other candle-derived files elsewhere in this repository are NOT traced
  by this section.
- Upstream contributors to the traced files up to that commit, by commit count on those paths in candle's
  history: Laurent Mazare (83), ivarflakstad (10), Lukas Kreussel (8), Eric Buehler (6), Nicolas Patry (2),
  Jon Craton (2), zmlcc, zachcp, Kyle Birnbaum, Anubhab Bandyopadhyay (1 each).
- Licence texts: below, verbatim from upstream at that commit, and in this repository's `LICENSE-MIT` /
  `LICENSE-APACHE` (shipped in every published crate, with this file).
- **Candle is used under its MIT licence** (the MIT choice of its dual MIT OR Apache-2.0 grant). MIT asks that the
  notice and licence text be included with copies; every published crate ships them (`LICENSE-MIT`, `LICENSE-APACHE`
  and this file). The Apache-2.0 requirement of a prominent notice in each changed file is therefore not relied on;
  in-file origin headers for the candle-derived files are scheduled (ROADMAP row 108), not skipped.
- **This repository began as a fork of candle. File-level tracing is in progress: a file that is not yet
  traced is listed as "not traced", never as ours.** The per-file table is `docs/provenance/` once it lands;
  until then only the files named above are traced. Both licences require their text and notice to travel
  with derived code. Candle's licence files name no copyright holder (the Apache appendix is the unfilled
  template); any notice that an individual candle file carries of its own is collected in the file-level
  pass, not here.

### candle `LICENSE-MIT`, verbatim at `549eacb011f25b2c6fc765d6c3dbd26e42f72ee6`

```text
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

### candle `LICENSE-APACHE`, verbatim at `549eacb011f25b2c6fc765d6c3dbd26e42f72ee6`

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## bitsandbytes

- Origin: https://github.com/bitsandbytes-foundation/bitsandbytes, `LICENSE` at commit `833649043474794b8fe7a4136e0c40faf077b2e0`.
- Used by: `fuel-tensor/src/nf4.rs` (the double-quantization absmax decode `code[q] * nested_absmax[i / nested_blocksize] + offset` and the packed-nibble order, re-implemented from `bitsandbytes/functional.py` and `bitsandbytes/backends/default/ops.py`; no source copied).

```text
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

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

## Qwen3-0.6B weights (test fixture)

- Origin: https://huggingface.co/Qwen/Qwen3-0.6B (Apache-2.0, Copyright 2024 Alibaba Cloud) and its bitsandbytes quantization https://huggingface.co/unsloth/Qwen3-0.6B-unsloth-bnb-4bit (apache-2.0).
- Used by: `fuel-tensor/tests/fixtures/bnb_double_quant/` (a 16-row excerpt of one weight matrix in quantized and original form, modified as listed in that directory's `PROVENANCE.md`). The licence text is `fuel-tensor/tests/fixtures/bnb_double_quant/LICENSE-Qwen3-0.6B-Apache-2.0.txt`.
