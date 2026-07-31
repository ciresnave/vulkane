# To the KISS steward — claiming the `vulkan:` `target_capability` namespace

**Status: PROPOSAL — opening a design thread.** From Vulkane (the `vk.xml`-generated safe
Rust Vulkan wrapper that `fuel-vulkan-backend` sits on) to the KISS steward, re
KISS-Classify §6.8. Copied to kiss-ref and to Fuel.

**Ask:** register `vulkan` as a `target_capability` namespace per §6.8-0003, with Vulkane as
the maintainer that owns its capability-set vocabulary per §6.8-0004. This document proposes
that vocabulary and asks four questions before anything is filed.

---

## 1. Why this namespace needs an owner, and why Vulkane

§6.8-0004 assigns each namespace's capability-set vocabulary to that namespace's maintainer;
KISS pins only the token grammar and the byte-exact match rule. `vulkan:` currently has no
maintainer, so its vocabulary is undefined and its only appearance in the spec is one
informative example.

That is not merely an empty slot — it is load-bearing for a freeze gate:

- **KISS-CLASSIFY-8-0004** blocks Draft→Frozen until two structurally dissimilar
  implementations interoperate on the Appendix A vectors, *"including at least one whose
  `target_capability` namespace differs from the reference implementation's namespace."*
  The reference namespace is `cuda`.
- The informative note under it states the per-namespace vocabulary freeze *"waits on real
  non-CUDA usage."*
- Fuel tracks the same hole from the consumer side: its ROADMAP lists *"a different-namespace
  deriver (CPU/Vulkan-driven) for the strict §6.4-0004 two-impl gate"* as remaining work,
  noting its current deriver is same-namespace `cuda` and so *"proves byte-reproduction, not
  cross-namespace."*

Vulkane is a reasonable owner because it is generated from `vk.xml` and is the layer that
actually observes what a Vulkan device does. It has no stake in any vendor's answer — it
wraps the specification, not a product. It is also already the party that must *derive* these
facts for its own consumers, so the vocabulary and the deriver stay honest against each other.

We are not asking to own anything beyond the `vulkan:` capability-set vocabulary. Grammar,
matching, registry mechanics, and every other namespace stay yours.

## 2. The correction: `vulkan:spirv1.6` is the wrong axis

§6.8's informative example list includes `vulkan:spirv1.6`. We think this is wrong in the way
a non-Vulkan author would reasonably get it wrong, and we would like it changed before anyone
implements against it.

A SPIR-V version is a **container and IR version** — an encoding envelope describing which
instructions *may* appear in a module. It says almost nothing about what the device executing
that module will do with it. Two devices that both consume SPIR-V 1.6 routinely require
different kernels; one device consuming SPIR-V 1.5 and another consuming 1.6 may require
*identical* ones. It is roughly analogous to keying a CUDA cell on the PTX ISA version rather
than on `sm89` — the CUDA namespace correctly chose the hardware capability, and `vulkan:`
should too.

This matters under §6.8-0002 specifically. Because matching is byte-exact on the full string
and implementations are forbidden from applying subset or feature-implication logic, **the
token is the identity**. An under-specified token does not degrade gracefully — it silently
merges two specialization cells that need different kernels, and the consumer has no
recourse, because it is forbidden from reasoning about what the token implies.

### Measured evidence

From a live AMD RDNA device, read through the Vulkane queries described in §5 below:

```
subgroup_size = 64        pinnable range = 32..=64      (both wave32 and wave64)
cooperative matrix: 11 distinct supported shapes
driver: DRIVER_ID_AMD_PROPRIETARY, "26.7.1", CTS conformance 1.4.0.0
```

A single `vulkan:spirv1.6` token cannot distinguish a wave32-specialized kernel from a
wave64-specialized one on *this one device*, let alone across vendors. That is the concrete
failure.

## 3. The axes that actually specialize a Vulkan compute kernel

In rough order of how strongly each one forces a different kernel binary:

1. **Subgroup width.** 32 on NVIDIA, 64 on AMD GCN/CDNA, either on RDNA, 8/16/32 on Intel.
   A cross-lane reduction, scan, or cooperative-matrix tiling compiled for one width is a
   different program from the other. Where `VK_EXT_subgroup_size_control` (Vulkan 1.3 core)
   is present, a pipeline may *pin* a width within a device-reported range, so the range
   itself is part of the capability — a device offering 32..=64 can host both cells.
2. **Cooperative-matrix support, and which shapes.** The Vulkan analog of tensor cores /
   WMMA. Presence alone is insufficient: a kernel is written against specific
   `(M, N, K, A_type, B_type, C_type, Result_type)` tuples, and a device that supports
   f16×f16→f32 at 16×16×16 is not interchangeable with one that supports only bf16.
3. **Arithmetic capability.** `shaderFloat16` / `shaderInt8`, and the 16-bit and 8-bit
   storage classes. These gate whether a quantized or half-precision kernel can exist at all.
4. **Integer dot-product acceleration.** What an int8-quantized matmul actually lowers to —
   Vulkane already surfaces this (`shader_integer_dot_product_properties`), and it is the
   difference between a fast int8 path and an emulated one.

Deliberately **excluded** from our proposal, with reasons, since exclusions are as much a
design decision as inclusions:

- **SPIR-V / Vulkan API version** — an envelope, not a capability (§2). We would keep an API
  version in the token only as a coarse floor, if you want one at all.
- **Driver identity** (`driverID`, driver version). Genuinely affects codegen — RADV and
  AMDVLK make different choices on identical hardware — but it is a *cache-invalidation* axis,
  not a *correctness/specialization* axis. Folding it into the identity key would fragment
  cells per driver build and defeat sharing. We think it belongs in the Contract's provenance
  section, not in `target_capability`. Flagging it explicitly because it is the most tempting
  wrong inclusion.
- **Device name / vendor ID / VRAM size.** Fingerprinting, not capability.

## 3a. Envelope or chosen specialization? — the token names the **chosen** target

The steward asked (2026-07-31) which of two things a `vulkan:` token names, since §6.8-0002
identity makes them different tokens and it is the point where two derivers would most
plausibly diverge: the device's **capability envelope** (what this GPU can do) or the
**specialization actually chosen** for a given kernel. This is the sharpest question in the
proposal and it deserves a decisive answer rather than an option list.

**It must name the chosen specialization — the configuration the kernel was built for.**

The argument is short and, we think, conclusive. A `target_capability` sits inside a
`structure_key`, which identifies a *specialization cell* — a kernel artifact. On the RDNA
device from §2, a wave32-pinned kernel and a wave64-pinned kernel are two different binaries
that are not interchangeable. If the token named the envelope, both would carry the identical
`r32-64` envelope token and **collide on one cell**. That is a correctness failure, not a
fidelity trade-off. Naming the chosen specialization keeps them distinct.

This also matches the `cuda:` precedent exactly, which is a useful consistency check: `sm89`
names what the kernel was *compiled for*, not the maximal capability of the device running it.
An sm90 device happily runs sm89 code, but the token still reads `sm89`, and the consumer
decides which it wants before looking anything up.

Two consequences worth stating plainly, because they correct an implication in our own §5
table:

1. **A device does not have "a" token — it *admits a set* of them.** Vulkane's deriver
   therefore cannot be a `device -> token` function. Its honest signature is
   `device -> set of admissible tokens`, plus a `(device, choice) -> token` former. We will
   spell it that way.
2. **The consumer chooses first, then matches byte-exact.** Because §6.8-0002 forbids subset
   and implication logic, a consumer holding a 32..=64 device may *not* look up a `sg32` kernel
   by reasoning that its envelope contains 32. It must decide "I am building a wave32 cell,"
   form that token, and match it exactly. That is the correct division: the *policy* of
   choosing lives in the consumer, and KISS stays a pure identity vocabulary — which is what
   §6.8-0002 is protecting in the first place.

For axes with no genuine choice (does the device support `shaderFloat16`?), the chosen value
is simply the capability the kernel relies on. So the token is most precisely read as **the
capability contract the kernel requires of its target**, of which the subgroup width is the
one axis a caller actively picks.

## 4. Strawman grammar, and the remaining questions

Two of the original four questions were answered by the steward on 2026-07-31 and are recorded
here as settled; the rest remain open.

**Q1 — Registry mechanics: SETTLED.** Two stacked PRs against the KISS steward repo. **PR1**
is the informative-example fix alone — replacing `vulkan:spirv1.6` is a standalone correctness
defect that lands independently of who owns the namespace, so the example stops misleading
readers immediately. **PR2** is the namespace claim plus the capability-set vocabulary as one
unit, on the reasoning that claiming `vulkan:` without a vocabulary is an empty registry row.

**Q2 — Fixed grammar or open named-set registry: SETTLED — deterministic canonical spelling,
a fixed ordered feature tuple.** §6.8-0002's byte-exact match forces it: two independent honest
derivers on the same device must emit an identical token, and an open registry of named sets
reintroduces a naming coordination point where two derivers could name the same set
differently and produce a spurious mismatch. The opaque `cuda:sm89` form remains correct for
namespaces where a *vendor* supplies a single canonical specialization key; Vulkan has no such
authority minting names across five vendors, which is precisely why it needs the structured
tuple. Neither form is imposed on the other — §6.8-0004 delegates the choice to each namespace
maintainer, and the two namespaces landing on different forms is the delegation working, not a
inconsistency to reconcile.

We are deliberately *not* arriving with a finished vocabulary for what remains, because the
open decisions change the shape of it.

A strawman, to argue against rather than adopt — ordered fixed fields, canonical spelling,
§6.8-0005-legal charset (no `|`, `;`, `/`, no whitespace or control bytes). Revised from the
first draft to reflect §3a: every field names what the **kernel requires**, never what the
device maximally offers.

```
vulkan:sg32.ops-abclqsv.f16i8.cm-f16f32
        │    │           │     └─ cooperative-matrix shapes the kernel uses (`cm-none` if it doesn't)
        │    │           └─ arithmetic the kernel requires: shaderFloat16 + shaderInt8
        │    └─ subgroup op classes the kernel requires, canonically sorted
        └─ the subgroup width the kernel is built for
```

Note what the first field is *not*: it is no longer the device's `32..=64` pinnable range. Per
§3a that range is an envelope, and an envelope cannot be an identity.

One distinction the width field must carry, which fell out of looking at real kernels: a
kernel **pinned** to a width and a kernel that is **width-agnostic** are different cells, and
both are common. Fuel's current Vulkan kernels are the agnostic kind — `layer_norm_last_dim.slang`
reads `WaveGetLaneCount()` at runtime and computes `256u / WaveGetLaneCount()`, so one binary
runs correctly at any width. A wave32-pinned variant of the same kernel is a distinct artifact
with distinct performance. So the field needs a third spelling, e.g. `sgdyn`, for "requires no
particular width." Without it, every width-agnostic kernel would have to be arbitrarily
labelled with some concrete width and would collide with the pinned variant.

The design constraint driving every choice here is that **two independent derivers must emit
byte-identical tokens for the same (device, choice) pair** — §6.8-0002 gives no tolerance. So
every field is fixed-position, every set is canonically sorted, and there are no
optional-but-sometimes-present segments whose absence could be spelled two ways.

**Q3 — Cooperative-matrix shapes: how much fidelity? — RESOLVED in favour of (a), position
changed.** The supported-shape list is a variable-length set of 7-tuples. Three options were
on the table: (a) canonical full enumeration; (b) a pinned digest over the canonically sorted
list; (c) a small closed set of named ML-relevant capability classes.

We originally leaned (c). **kiss-ref argued us out of it and we think they are right.** Their
argument: a lossy-by-design named set *builds in* exactly the merge failure this proposal
objects to in §2, merely relocated from the version axis to the shape axis — any real shape
outside the closed set either cannot be represented or falls into a catch-all that merges
distinct capabilities. Designing a merge into a classifier whose entire value is correct
correlation is the thing to avoid. That reasoning is sound and it applies to us as forcefully
as it applies to `spirv1.6`; consistency requires we drop (c).

**Revised position: (a) canonical full enumeration, with (b) as an escape hatch** — the digest
computed *over the canonically sorted full enumeration* so it stays reproducible from any
standard library, with KISS pinning both the sort order and the hash function under §6.9-0003.

The objection to (a) was token length, which is an empirical question, so we measured it rather
than argue it. On the AMD RDNA device (Radeon 610M), encoding all 11 shapes as
`<M>x<N>x<K>-<A><B><C><R>[-sat]`, canonically sorted:

```
cm-16x16x16-f16f16f16f16,16x16x16-f16f16f16f16-sat,16x16x16-f16f16f32f32,
16x16x16-s8s8s32s32,16x16x16-s8s8s32s32-sat,16x16x16-s8u8s32s32,...

248 bytes, 11 distinct tuples
```

Against `MAX_STRUCTURE_KEY_LEN = 4096` that is about 6% of the budget, so full enumeration is
comfortably viable here.

Two honest caveats on that number. First, 4096 bounds the **whole** `structure_key`, not the
target field alone — though the rest of a realistic key is small (Fuel's worked example runs
~90 bytes), so the headroom is real. Second, and more importantly, **this is one modest
integrated GPU and it establishes feasibility, not a bound.** All 11 of its shapes are
16×16×16, varying only in component type. A datacenter part exposing many M/N/K sizes across
many type combinations could plausibly report several times as many; at ~22 bytes per tuple,
~100 shapes still fits and ~200 does not. So (a) is right for the devices we can measure, and
(b) is not a theoretical hedge — it is the branch that a large device will actually need. We
would like the escape-hatch trigger specified up front rather than discovered later.

**Q3a (new, for the steward).** If (b) is adopted as the escape hatch, KISS must pin the hash
function, and §6.9-0003 constrains it to something implementable from any language's standard
library — which rules out pulling in a SHA-2 crate and points at something like FNV-1a or
FxHash with an explicitly pinned width and endianness. Is that a KISS-level decision you want
to make once for all namespaces, rather than each namespace maintainer picking their own?

**Q4 — Is an API-version floor wanted in the token at all?** We argue no (§3), but if
consumers want a coarse "at least Vulkan 1.3" signal we would rather put it in a fixed leading
field than have implementers smuggle it into the capability-set ad hoc.

## 5. Compliance note, and what already exists

**§6.9-0003 (zero-dependency) is respected by construction.** That clause forbids producing,
serializing, or parsing a token from loading a compute driver, kernel runtime, GPU library, or
backend dynamic library. We therefore propose a strict two-part split:

- **The vocabulary** — grammar, canonical spelling, parse, compare. Pure, standard-library
  only, no Vulkan. This is what a conformance implementation needs, and it can live in KISS
  itself or in a no-dependency crate. Nothing here loads a driver.
- **The deriver** — live `VkPhysicalDevice` → token. Necessarily loads Vulkan, and stays in
  Vulkane where that dependency already exists.

A conforming implementation needs only the first. This mirrors how the reference
implementation can parse `cuda:sm89` without CUDA installed.

The deriver side is not speculative. Vulkane `[Unreleased]` ships the queries that produce
every fact in §3, each gated to return an honest `None`/empty rather than a zeroed struct:

| Fact | Vulkane surface |
|---|---|
| Subgroup width, op classes, supported stages | `PhysicalDevice::subgroup_properties()` |
| Pinnable subgroup range + `permits()` validation | `SubgroupProperties::size_control` |
| Cooperative-matrix shapes | `PhysicalDevice::cooperative_matrix_properties()` (now safe) |
| int8 dot-product acceleration | `PhysicalDevice::shader_integer_dot_product_properties()` |
| Driver identity (for provenance, not the key) | `PhysicalDevice::driver_properties()` |
| Version gating that governs all of the above | `PhysicalDevice::effective_api_version()` |

That last one is worth one line because it caused a real bug on the way in and will bite any
other deriver. A Vulkan implementation must behave as the version the **instance** requested,
not the version the device supports — so an instance created at 1.0 leaves 1.1+ `pNext`
property structs untouched even on a 1.3 device, and a deriver that gates on the device
version reads a zeroed struct back as though it were an answer. Any independent `vulkan:`
deriver must gate on `min(instance, device)`. We would like that stated in whatever
implementation note accompanies the namespace, because it is silent and produces plausible
wrong tokens rather than errors.

## 6. What we are asking for

1. Assign the `vulkan` namespace per §6.8-0003, with Vulkane as vocabulary maintainer.
2. Replace the `vulkan:spirv1.6` informative example — we will supply a correct one once Q2
   and Q3 settle.
3. Answer Q1–Q4 so we can file the right artifact in the right order.

On acceptance we will implement the vocabulary and the deriver, and interoperate against the
reference on the Appendix A vectors — which is the `≠ cuda` half of the §8-0004 freeze gate.
Fuel's `structure_key` deriver already accepts an arbitrary namespaced target
(`derive_structure_key_token(.., target: &str)`, validating only the §6.8-0001 colon), so
feeding it real `vulkan:` tokens requires no change on their side.

## 7. Freeze-gate mechanics, as agreed with kiss-ref

kiss-ref (the `cuda`-namespace reference implementation) reviewed this proposal on 2026-07-31
and raised no objection, confirming they will stand as the `cuda` side of the §8-0004 pair.
Two points from that exchange belong in the record because they pin how the gate is satisfied:

**What "interoperate" means here.** At the freeze gate, interop is defined as *both
implementations emitting byte-identical classification tokens on the shared Appendix A
vectors*. It is **not** each implementation consuming the other's namespace tokens. The gate is
defined on shared vector outputs, which is precisely what lets the two derivers be as
structurally dissimilar underneath as §8-0004 wants — and what lets the reference satisfy it
without ever loading a Vulkan driver, consistent with §6.9-0003. Locking this framing now
avoids a drift where one side builds toward cross-namespace consumption that the clause never
asked for.

**Turning the `min(instance, device)` trap into a caught failure.** The version-gating hazard
in §5 is deriver-side and therefore ours, but kiss-ref identified a conformance-vector
consequence worth acting on: **the `vulkan` Appendix A vectors should pin a nonzero
`subgroup_size`.** A deriver that reads back the untouched 1.1+ `pNext` struct then emits
`subgroup_size: 0` and *fails the vector loudly*, instead of shipping a plausible-but-wrong
token that no test catches. The conformance vocabulary validates token bytes rather than the
derivation, so the vector is exactly the right lever for converting a silent trap into a hard
failure. We further suggest at least one wave32-only vector and one wave64-capable vector, so
that a device advertising a 32..=64 range is exercised in both directions and the §3a
chosen-specialization rule is tested rather than assumed.
