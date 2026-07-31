# kiss-vulkan-vocab

The **`vulkan:` capability-set vocabulary** for [KISS-Classify][kiss] §6.8
`target_capability` tokens — canonical spelling, parsing, and byte-exact
comparison.

[kiss]: https://github.com/ThinkersJournal/KISS

```rust
use kiss_vulkan_vocab::{VulkanTarget, Subgroup, OpClasses, Arith, CoopMatrix};

let t = VulkanTarget {
    subgroup: Subgroup::Fixed(32),
    ops: OpClasses::BASIC | OpClasses::ARITHMETIC,
    arith: Arith::FLOAT16,
    coop: CoopMatrix::None,
};
assert_eq!(t.to_token(), "vulkan:sg32.ops-ab.arith-f16.cm-none");
assert_eq!(VulkanTarget::parse(&t.to_token()).unwrap(), t);
```

## Zero dependencies, on purpose

KISS-CLASSIFY-6.9-0003 requires that producing, serializing, or parsing a
`target_capability` token need no compute driver, kernel runtime, GPU library,
or backend dynamic library — an implementation must manage with its language's
standard library alone, and the reference implementation holds no exemption.

So this crate has **no dependencies and no Vulkan linkage**, and a test reads
the manifest on every build to keep it that way. Deriving a token from a real
`VkPhysicalDevice` obviously *does* need Vulkan; that lives in
[`vulkane::kiss`](https://docs.rs/vulkane) behind its `kiss-target` feature.
A conformance implementation needs only this half.

## The token names a chosen specialization, not a device

A `target_capability` sits inside a `structure_key`, which identifies a
specialization **cell** — a kernel artifact. A wave32-pinned kernel and a
wave64-pinned kernel are different binaries, so they must be different tokens;
a token naming the device's capability *envelope* would collide them onto one
cell. It follows that **a device admits a set of tokens rather than having
one**, and the consumer chooses before matching.

This mirrors `cuda:sm89`, which names what a kernel was compiled *for*, not the
maximum capability of the part running it.

Because §6.8-0002 matching is byte-exact and forbids subset or
feature-implication logic, a consumer holding a device that supports widths
32..=64 may **not** look up a `sg32` kernel by reasoning that its envelope
contains 32. It must decide it is building a wave32 cell, spell that token, and
match it exactly. Choice policy lives in the consumer; this crate is a pure
identity vocabulary.

## Grammar

```text
vulkan:<subgroup>.<ops>.<arith>.<coop>
```

Four fixed-position fields separated by `.`; parts within a field by `-`.

| Field | Examples | Meaning |
|---|---|---|
| subgroup | `sg32`, `sg64`, `sgdyn` | the width the kernel is built for, or `sgdyn` for width-agnostic |
| ops | `ops-abr`, `ops-none` | subgroup operation classes required, canonically sorted |
| arith | `arith-f16-i8`, `arith-none` | arithmetic capabilities required |
| coop | `cm-16-16-16-f16-f16-f32-f32`, `cm-none`, `cm-fnv1a64.<hex>` | cooperative-matrix shapes used |

Two spellings that differ by any byte are different cells, so the crate
**rejects** legal-but-non-canonical input (unsorted op letters, duplicated
shapes, leading zeros, uppercase digest hex) rather than normalizing it.
Accepting two spellings of one target would let them silently fail to match
each other, which is worse than accepting neither.

`sgdyn` exists because a kernel compiled for a fixed width and one that reads
`gl_SubgroupSize` / `WaveGetLaneCount()` at runtime are different artifacts with
different performance, and both are common. Without a distinct spelling every
width-agnostic kernel would be labelled with an arbitrary concrete width and
collide with the pinned variant of itself.

## Status

Pre-1.0, tracking a **draft** standard. KISS is pre-1.0 and explicitly
unfrozen; the `vulkan:` vocabulary is proposed, not ratified. The normative
definition is the KISS spec text under §6.8-0004 — this crate is the reference
implementation and must match it byte for byte. **The spec wins on any
disagreement.**

## License

MIT OR Apache-2.0.
