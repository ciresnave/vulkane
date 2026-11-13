kind: Added `[architecture]` declaration in `.coderipper.toml` (ROADMAP row 9c, docs/config only)
---
Declares the 13 workspace crates as components in two layers (12 leaf crates below, `vulkane` on top). ARC-001 and ARC-002 now reach a verdict instead of "not run": 3 edges judged, 0 findings, 0 cycles (coderipper 0.43.x, `--profile extended`, at this branch). ARC-003/004/006/007/012 stay "not run": they need policy/volatile/composition/budget declarations that this PR does not invent; they follow with row 9 (module groups inside `vulkane`, where the crate::safe cycle lives). No code changed.
