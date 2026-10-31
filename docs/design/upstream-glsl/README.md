# tree-sitter-glsl: upstream contribution, ready to post (not posted)

Two items for `tree-sitter-grammars/tree-sitter-glsl`, prepared and verified
here, **not yet posted**. CireSnave approved sending both (board 352 A); the
PM account posts them.

| File | What it is |
|---|---|
| `qualifier-only-declarations.patch` | The change: `grammar.js`, one corpus test, `src/grammar.json`. 3.4 KB. |
| `pr-qualifier-only-declarations.md` | PR title and body for that change. |
| `issue-parenthesised-constructor-call.md` | Separate issue: a second, unfixed gap, with repro and control. |

## Why `src/parser.c` is not in this directory

Upstream commits its generated `src/`. The regenerated `src/parser.c` changes
by about 228,000 lines (state numbers shift); the full patch is 9.7 MB, which is
larger than this whole repository. So `parser.c` is regenerated, not stored:
generation is deterministic (run twice, identical bytes) and the expected hashes
are below. `src/node-types.json` is unchanged by the patch.

## Versions everything was run with

- Upstream: `tree-sitter-grammars/tree-sitter-glsl`, default branch `master`,
  commit `24a6c8ef698e4480fecf8340d771fbcb5de8fbb4` (2025-03-16), package
  version 0.2.0. Its `grammar.js` is byte-identical (after line-ending
  normalisation) to the published `tree-sitter-glsl` 0.2.0 crate.
- `tree-sitter-cli` **0.24.7**, built from crates.io (`cargo install
  tree-sitter-cli --version =0.24.7 --locked`), generating with `--abi 14` to
  match upstream's committed `LANGUAGE_VERSION 14`. (0.27.1 was used earlier for
  the measurements in `../glsl-parser-emitter.md`; it is not used for anything
  below.)
- `tree-sitter-c` **0.24.2** from crates.io, `grammar.js` only, placed at
  `node_modules/tree-sitter-c/grammar.js`. Upstream's `package.json` instead
  names `tree-sitter-c` by git URL (unpinned), so it is **not established** that
  0.24.2 is the base upstream last generated against; the control below is what
  supports it.
- Nothing else from upstream's `devDependencies` was installed.

## Control: the toolchain reproduces upstream before any change

With the versions above, `tree-sitter generate --abi 14` on **unmodified**
upstream `master` leaves `src/parser.c`, `src/grammar.json` and
`src/node-types.json` byte-identical to the committed ones (`git diff` empty,
files confirmed rewritten by the run). So a diff after the patch is the patch's
doing, not the toolchain's.

## Reproduce

```sh
git clone https://github.com/tree-sitter-grammars/tree-sitter-glsl
cd tree-sitter-glsl && git checkout 24a6c8ef698e4480fecf8340d771fbcb5de8fbb4
mkdir -p node_modules/tree-sitter-c          # then put tree-sitter-c 0.24.2's grammar.js there
git apply qualifier-only-declarations.patch
tree-sitter generate --abi 14                 # CLI 0.24.7
tree-sitter test                              # 14 passed, 0 failed
sha256sum src/parser.c src/grammar.json src/node-types.json
```

Expected SHA-256 after the patch and regeneration:

```
2ca2f818b6acde7172679f20867014bd4fcddbff7abcf83cd46267f65896aa95  src/parser.c
c57051d849abb0cc771b7d66293d0cafe6ed1b2a6d660c65142ff3b7491155fd  src/grammar.json
a51e4f59e6a530af6e3695ab7fcab7caa7c5b244a71ff3e807bfda74c1b4dd2c  src/node-types.json
```

Unmodified upstream `src/parser.c`, for comparison:
`a175dc73eab717b1b745a6d1826ac1ddef1a46c72dc174f07ec721ac1f32e9ad`.

(`tree-sitter test` on Windows also leaves `parser.exp`, `parser.lib` and
`parser.obj` in the working directory; do not commit them.)

## What was measured

- New corpus test `Qualifier-only declarations`: **fails before** the patch
  (exit 1, `ERROR` where `declaration` is expected) and **passes after**. The 13
  existing corpus tests pass before and after: 14 passed, 0 failed.
- 27 compute shaders (`*.glsl`) from a downstream project: **0 of 27** parse
  without an `ERROR` or `MISSING` node on upstream `master`; **24 of 27** with
  the patch. The remaining 3 all contain a parenthesised constructor call and
  are the separate issue. This is a count of files parsed without error, not a
  check that the trees are right.
- The gap-2 repro fails identically with and without the patch (the issue is
  independent of the PR).

Not verified: behaviour with `tree-sitter` CLI versions other than 0.24.7,
`tree-sitter-c` other than 0.24.2, or the `bindings/` builds (only
`tree-sitter generate` and `tree-sitter test` were run). Upstream's own CI
(`tree-sitter/parser-test-action`) will exercise the bindings.

## Contribution rules

Upstream has no `CONTRIBUTING` file and no CLA; licence is MIT. Its
`update-cpp-parser.yml` workflow regenerates on a schedule, which may
regenerate `src/` independently of this PR.
