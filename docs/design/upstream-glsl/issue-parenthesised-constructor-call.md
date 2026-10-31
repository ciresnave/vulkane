# Issue

**Title:** Parenthesised expression starting with a constructor call, e.g. `(float(1) - 1.0)`, produces an ERROR

**Body:**

A parenthesised expression whose first operand is a type-constructor call fails
to parse. The same expression with the operands swapped, or without the
parentheses, parses cleanly.

Failing:

```glsl
void main() {
    float s = (float(1) - 1.0);
}
```

Controls (same file shape, no error):

```glsl
float s = (1.0 - float(1));   // operands swapped
float s = float(1) - 1.0;     // no parentheses
```

Result for the failing input:

```
(init_declarator
  declarator: (identifier)
  value: (binary_expression
    left: (parenthesized_expression
      (ERROR ...
```

Both controls parse without an error. Versions: `tree-sitter-glsl` 0.2.0 / `master` at
`24a6c8ef698e4480fecf8340d771fbcb5de8fbb4`, regenerated with `tree-sitter-cli`
0.24.7 and `tree-sitter-c` 0.24.2 (this reproduces the committed `src/` exactly);
the same input also errors with `tree-sitter` CLI 0.27.1. It is independent of
the qualifier-only-declarations change: it fails identically with and without it.

### What I tried (none worked)

These were tried with `tree-sitter` CLI 0.27.1 on a set of 29 compute shaders:

- Adding a `call_expression` alternative whose callee is `primitive_type`. It
  fixes the three files that hit this, but regresses five others (files using
  cooperative-matrix code), so I did not propose it.
- Giving that alternative `prec(0)` or `prec.dynamic(-10)`. Generation fails
  with `Unresolved conflict ... primitive_type • '('`.

An earlier guess of mine, that the cause is C's `cast_expression`, was wrong:
removing it from the grammar changes nothing for this input.

I do not have a fix. This is a report with a minimal repro in case it is quick
for someone who knows the conflict structure.
