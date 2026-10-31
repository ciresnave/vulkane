# PR

**Title:** Parse qualifier-only declarations such as `layout(local_size_x = 64) in;`

**Body:**

A GLSL declaration can consist of qualifiers alone, with no type and no name.
The common one is the compute-shader workgroup size:

```glsl
layout(local_size_x = 64) in;
```

The same form is used for `layout(std430) buffer;` and
`layout(binding = 0) uniform;`. On current `master` these produce an `ERROR`
node, and the error can swallow the declarations that follow it (in a four-line
file of such declarations, one `ERROR` spans all four lines).

```
$ tree-sitter parse repro.comp      # repro.comp = the one line above
(translation_unit
  (ERROR
    (layout_specification ...)
    (ERROR)))
```

### Change

In the `declaration` rule, add one alternative: one or more of the storage
qualifiers or a `layout_specification`, then `;`.

```js
seq(repeat1(choice('in', 'out', 'uniform', 'buffer', 'shared', $.layout_specification)), ";"),
```

`src/grammar.json` and `src/parser.c` are regenerated. `src/node-types.json` is
unchanged.

### Test

New corpus entry `Qualifier-only declarations` in `test/corpus/basic.txt`
(four declarations). It fails without the grammar change and passes with it. The
13 existing corpus tests are unaffected: 14 passed, 0 failed.

### How it was generated

`tree-sitter-cli` 0.24.7 with `tree-sitter generate --abi 14`, and
`tree-sitter-c` 0.24.2. On unmodified `master` this toolchain reproduces the
committed `src/` byte for byte, so the `src/` diff here is only the effect of
the change. `parser.c` is large because state numbers shift, as usual for a
generated parser.

### Scope

This does not touch the separate problem of parenthesised constructor calls such
as `(float(1) - 1.0)`; that is filed as its own issue.

Measured on 27 compute shaders from a downstream project: 0 of 27 parsed
without error before, 24 of 27 after. The other 3 are the constructor-call
issue.
