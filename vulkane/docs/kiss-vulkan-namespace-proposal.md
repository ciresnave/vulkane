> # ⚠ SUPERSEDED — HISTORICAL RECORD, NOT A SPECIFICATION
>
> **This document asked for something that was granted. It is kept for the
> reasoning and evidence that won the namespace, and for nothing else.**
>
> **The normative definition of `vulkan:` is KISS's `spec/namespaces/vulkan.md`.**
> Per KISS-Classify §6.8-0003 the registry's `vocabulary` field names where a vocabulary is
> *normatively defined*, and it names that file. Nothing in this repository is
> normative for the namespace, this document least of all.
>
> **Do not read a grammar out of it.** This document's §4 proposes `vulkan:<fields>.fnv1a64:<hex>`.
> The ratified grammar, reproduced BYTE-FOR-BYTE from the `grammar` field of
> `kiss-vulkan-vocab/manifest/vulkan-vocabulary.json`, which is authoritative:
>
> ```
> vulkan:<subgroup>.<ops>.<arith>.<coop>.<coopvec>
> ```
>
> That block holds the grammar and nothing else — no version suffix, no trailing
> commentary — because this is a document about BYTE-EXACT matching and a reader
> who copies the line must get a usable value. It is vocabulary_version 5 as of
> writing, and the version is stated here in prose rather than inside the block
> for that reason. `vulkane/tests/superseded_doc_grammar.rs` asserts the two are
> byte-identical, so this copy cannot drift from the manifest the way the count
> in KISS's registry row did.
>
> Those are different token shapes, not a refinement of one into the other. KISS-Classify §6.8-0002
> matching is **byte-exact**, so anything emitted from that strawman matches nothing
> any current implementation produces. The machine-readable vocabulary lives at
> `kiss-vulkan-vocab/manifest/vulkan-vocabulary.json` and is regenerated and
> byte-compared on every CI run; **that file, not this one, is what a tool should read.**
>
> **Statements below that were true on 2026-07-31 and are false now**, listed because
> a reader landing mid-document will not see this header:
>
> - *"`vulkan:` currently has no maintainer, so its vocabulary is undefined"* —
>   Vulkane is the registered maintainer and the vocabulary is at version 5.
> - *"asks four questions before anything is filed"* — all four were answered and
>   the namespace was filed, registered, and revised four times since.
>
> Kept rather than deleted because the argument for *why* the namespace needed an
> owner, and Fuel's two-device measurement, exist nowhere else. **The history is the
> value; the vocabulary in it is four revisions stale.**

---

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

**(a) Two devices at different fixed widths, independently measured.** Produced by the Fuel
session (cited with permission) from `fuel-vulkan-backend`'s device probe built against
vulkane `v0.9.0` — a separate implementation, on different silicon, reached through the same
public API:

```
AMD Radeon(TM) 610M             subgroup_width=64  DRIVER_ID_AMD_PROPRIETARY
                                "AMD proprietary driver 26.7.1 (AMD proprietary shader compiler)"
NVIDIA GeForce RTX 4070 Laptop  subgroup_width=32  DRIVER_ID_NVIDIA_PROPRIETARY
                                "NVIDIA 610.88"
```

Both enumerated by one `vkEnumeratePhysicalDevices` call on one machine, instance at
`ApiVersion::V1_2`. `size_control` was `None` on both, consistent with the 1.3 gate described
in §5 — so these are the devices' **default** widths, not pinned ones, and should not be read
as chosen specializations in the §3a sense.

Three things make this the primary citation, and it leads the argument on the steward's
advice (2026-07-31):

1. **It is an independent measurement by a consumer, not the author.** §8-0004 asks for real
   non-CUDA usage; this is a downstream project reporting the fact through the public API,
   which is materially different evidence from the API's own maintainer reporting it.
2. **It separates on two orthogonal axes, not one.** The devices differ in wave width (64 vs 32)
   *and* in ICD. Those are independent: an AMD device under RADV versus AMDVLK would share a
   width while differing in codegen. A single encoding-envelope token encodes neither axis, let
   alone their product — so the `spirv1.6` failure is not merely width-shaped.
3. **This is an unremarkable consumer laptop** — an integrated 610M alongside a mobile 4070 —
   not a curated test rig. That the divergence shows up *by accident* on ordinary hardware is a
   stronger argument than a deliberately assembled pair would be, which invites the objection
   that the counterexample was constructed.

**(b) Corroborating: one device admitting two widths.** From a live AMD RDNA device, read
through the Vulkane queries described in §5:

```
subgroup_size = 64        pinnable range = 32..=64      (both wave32 and wave64)
cooperative matrix: 11 distinct supported shapes
driver: DRIVER_ID_AMD_PROPRIETARY, "26.7.1", CTS conformance 1.4.0.0
```

This adds the case (a) cannot show: a *single* device that admits both wave32 and wave64, so
the two specializations are not merely different hardware but different admissible points on
one GPU. A single `vulkan:spirv1.6` token cannot distinguish them there either — which is why
§3a resolves the token to name the chosen specialization rather than the envelope.

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

The grammar below now reflects the steward's Q3a/Q3b rulings (2026-07-31), which pin a
**two-level delimiter discipline**: exactly one `:` (the namespace separator), `.` between
fields, and `-` within a tuple — with an explicit separator between adjacent dtype tokens
**required**, juxtaposition forbidden. Exact spelling within those constraints is ours as the
§6.8-0004 namespace owner. Every field names what the **kernel requires**, never what the
device maximally offers (§3a).

```
vulkan:sg32.ops-abclqsv.f16i8.cm-16-16-16-f16-f16-f32-f32
        │    │           │     └─ cooperative-matrix shapes the kernel uses (`cm-none` if it doesn't)
        │    │           └─ arithmetic the kernel requires: shaderFloat16 + shaderInt8
        │    └─ subgroup op classes the kernel requires, canonically sorted
        └─ the subgroup width the kernel is built for (`sgdyn` if width-agnostic)
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

**Q3a — RULED (steward, 2026-07-31): pin the hash once at KISS level, not per-namespace.**
FNV-1a, 64-bit, offset-basis and prime pinned as spec literals, computed over the canonical
enumeration byte-serialization, emitted as fixed-width lowercase hex with byte order pinned.
Stdlib-trivial, so §6.9-0003 holds; derivers are non-adversarial, so accidental-collision
resistance is the only requirement and no SHA is needed. The digest carries an
algorithm-marker prefix so a future hash migration produces distinguishable rather than
silently colliding tokens, and the full-vs-digest choice is **length-triggered, never a
deriver preference**. The reasoning below is retained because it is what the ruling settles.

> **Defect found in the ruling as first spelled — CORRECTED by the steward same day.** The
> original marker was `fnv1a64:`, which would make a digest-form token
> `vulkan:<fields>.fnv1a64:<hex>` — **two** colons. §6.8-0001 requires *exactly one* and states
> that a reader MUST reject a token with "zero or more than one `:`" via typed decline, so the
> digest branch would have been unusable by every conforming reader — and unusable specifically
> in the large-device case that is hardest to test, so it could have shipped looking fine. The
> intent was right and is retained: a marker keeping retired and current digests
> distinguishable is §3.3's burn-the-retired-ID discipline applied to the hash primitive.
> **Corrected marker: `fnv1a64.<hex>`** — the single `:` stays the namespace separator and
> everything below it uses `.` / `-`, matching the Q3b delimiter discipline.

The bundle the ruling pins, stated for implementers:

1. **The trigger** — a deterministic length test on the *encoded* field, e.g. "if the canonical
   enumeration string exceeds N bytes, hash instead." Not a tuple count, not an
   implementation-defined budget; a byte length on the produced string.
2. **The hash input** — the hash MUST run over *that same canonically-sorted enumeration
   string*, never over the raw tuple set. This is what keeps (a) and (b) faithful to one form:
   the switch becomes a pure length-driven representation swap with identical input semantics,
   so a deriver can disagree only about *whether* to hash, never about *what* is being hashed.
3. **The function** — §6.9-0003's stdlib-only rule rules out reaching for SHA-2, pointing at
   something like FNV-1a with explicitly pinned width and endianness.

Our ask: pin all three **once, for all namespaces**, rather than letting each namespace
maintainer choose. Divergent per-namespace hashes or thresholds are exactly the kind of thing
that is invisible until two implementations meet.

**Q3b — pin the intra-field separator, and pin unique-decodability as a standing §6.1
constraint.** Our strawman writes a component-type tuple by juxtaposition
(`f16f16f32f32`), which is only safe if the §6.1 dtype tokens are uniquely decodable under
concatenation. We checked rather than assumed, and the result argues against juxtaposition
more strongly than we expected:

- The 22-token set is **not prefix-free**. There are two violations today:
  `e4m3fn` is a prefix of `e4m3fnuz`, and `e5m2` is a prefix of `e5m2fnuz`.
- It nonetheless **is** uniquely decodable — Sardinas–Patterson terminates with no dangling
  suffix that is itself a codeword, and an exhaustive check finds zero ambiguous 2-tuples.

So juxtaposition happens to work today, but it rests on the *global* unique-decodability
property (which requires Sardinas–Patterson to verify) rather than the *local*, eyeball-checkable
prefix property — and the set already violates the easy property, which means nobody is
maintaining even the informal version of this invariant. The §6.1 token set is documented as
growing ("when the spec changes, this binding follows"), and the only property §6.1 currently
pins over its tokens is **distinctness**, which is strictly weaker than unique decodability
(`{a, ab, b}` are distinct yet `ab` parses two ways).

The failure is one token away and non-local: adding `uz` to the set would immediately give
`e4m3fnuz` two parses (`e4m3fnuz` and `e4m3fn` + `uz`), silently mis-parsing every juxtaposed
tuple that contains it. A reviewer approving a new dtype has no reason to run a decodability
check, because nothing says they must.

**Q3b — RULED (steward, 2026-07-31), both parts, harder than we asked for:**

1. **The explicit intra-tuple separator is NORMATIVE — juxtaposition is forbidden.** The
   reasoning: a property nobody can check by eye, and that nobody is currently maintaining, is
   not a property to build a wire identity on. Decoupling tuple parsing from §6.1's evolution
   is the point. Two-level discipline pinned: `.` between fields, `-` within a tuple; the
   single `:` remains the namespace separator.
2. **Unique decodability becomes a machine-enforced §6.1 constraint, not prose.** The steward's
   argument is the sharper version of ours: "a reviewer has no reason to run a decodability
   check because nothing says they must" is exactly why the prefix property already rotted, and
   *a second prose rule rots the same way*. So a Sardinas–Patterson check goes into the §6.1
   dtype-vocab test suite and **fails CI** when a new token breaks unique decodability. The
   steward owns landing it, as KISS-Classify infrastructure rather than Vulkan-specific work.
   With (1) in force for `vulkan:` this is belt-and-suspenders here, but it protects any other
   encoding that leans on the property.

**Measured cost of the separator, re-run against the mandated spelling** (we had estimated this
badly the first time and the estimate propagated, so it is worth stating precisely):

| Encoding | Bytes | Share of `MAX_STRUCTURE_KEY_LEN` |
|---|---|---|
| Juxtaposed (`16x16x16-f16f16f32f32`) | 248 | 6.1% |
| Separated (`16-16-16-f16-f16-f32-f32`) | **281** | **6.9%** |

The separator costs **+33 bytes, +13.3% on the field** — not the "~6%" figure quoted earlier in
this document and repeated back in review, which was the field's share of the total budget, a
different quantity. The conclusion is unchanged and if anything better supported: a 13.3%
increase on a field occupying 6.9% of the budget is negligible, and it leaves headroom for
roughly **163 tuples** at this tuple width before the digest escape hatch is needed.

We deliberately did not mint the separator ourselves. kiss-ref confirmed (2026-07-31) that the
reference implementation does not yet implement the §6.7 structure-key codec —
`kiss-classify-vocab` is the §6.1 dtype vocabulary only — so there was no `cuda`-side
convention to inherit, and inventing one per-namespace is precisely the divergence this
proposal exists to prevent.

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

**Which vectors actually carry the interop obligation.** That statement splits in two, and
only one half is shared:

- **Codec-neutral vectors** — envelope, field order, separator choice, escaping of the
  type-tuple string, dedup and sort order, `MAX_STRUCTURE_KEY_LEN` discipline. **Both
  implementations must encode these byte-identically regardless of namespace.** This is the
  real interop surface, and it is what kiss-ref will diff against the `cuda` side's structure-key
  encoding.
- **Capability-value vectors** — `sm89` versus `cm-16x16x16-...`. Namespace-specific; each
  implementation owns its own and they are not cross-emitted.

If the codec-neutral layer matches byte-for-byte and only the payloads differ, the shared codec
is proven identical, which is what §8-0004 should be testing. We will label each drafted
`vulkan` vector as codec-neutral or capability-specific when we send them.

**The `(device, choice) -> token` deriver shape is shared, not a Vulkan quirk.** kiss-ref
confirmed §3a's resolution holds on the `cuda` side too: a physical Ada part runs kernels built
for `sm_80` / `sm_86` / `sm_89` via binary and PTX-JIT compatibility, so a CUDA device likewise
*admits a set* of tokens rather than having one. Both namespaces therefore share the same
deriver shape while remaining structurally dissimilar underneath — which is a stronger position
for the §8-0004 pair than a coincidence would be, and worth recording as a shared property of
the descriptor rather than a per-namespace decision.

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
