# Third-party notices

**No third-party source is copied into this crate.** Its structure (a `Backend` over `lower_dag` with its own spellers) follows
the published `fishstringer-slang-emit` (`MIT OR Apache-2.0`), which this crate does not depend on; the spellers, tables and
harness are written for WGSL.

Dependencies (not copied or redistributed): `unpopped`, `unpopped-vocab` (`MIT OR Apache-2.0`); dev only: `naga`
(`MIT OR Apache-2.0`).

Every source file declares `SPDX-License-Identifier: MIT OR Apache-2.0`.
