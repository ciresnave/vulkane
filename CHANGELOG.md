# Changelog

All notable changes to vulkane will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — clause citations in `kiss-vulkan-vocab` name their spec

Every `§` citation in the exported crate now reads `KISS-Classify §…` rather than a bare
clause id. **A bare id does not identify a clause.** `6.8-0002` is *defined* in at least three
KISS specs and means something different in each — token matching in Classify, the ULP oracle
in Conform, a typed decline in Emit. So a reader resolving a bare cite can land on a real
clause that says something unrelated, which is worse than failing to find one.

**Two of the twenty-three were in `Display` output, not comments**, and those are the ones
that mattered most: an error message is where a reader with no context meets a citation.
The `MissingColon` and forbidden-byte messages now name the spec. No test asserted on either
string, and the emitted manifest is byte-identical.

Scoped to the exported crate deliberately. The superseded namespace proposal keeps its bare
cites — it already carries a banner saying not to build against it — and **the CHANGELOG's own
historical entries are left verbatim**, because a changelog records what was said at the time
and correcting a citation inside one rewrites history rather than fixing a document.

### Changed (breaking) — `vulkan:` vocabulary version 5 names three arithmetic capabilities

`<arith>` gains `i16`, `i64` and `f64`, naming the core `VkPhysicalDeviceFeatures` bits
`shaderInt16`, `shaderInt64` and `shaderFloat64`. `Arith` gains `INT16`, `INT64` and `FLOAT64`,
and `vulkane::kiss` derives all three from `supported_features()`.

**This changes the bytes of almost every token in the wild.** Unlike FP8 or the packed types, these
are not exotic — an RTX 4070 and an AMD Radeon 610M both report all three, so a token derived from
either changes. §6.8-0002 matching is byte-exact, so a version-4 token and its version-5 equivalent
**do not match**: caches keyed on version-4 tokens are stale and must be **invalidated, not
migrated**. A device reporting none of the three is unaffected.

That this bumps rather than adds is the §4 additive test run mechanically: all three are core
feature bits assigned long before the registry baseline `VK_HEADER_VERSION` 348, so a conformant
device could always have reported them. Version 5's own baseline is 348. They land together because
each would otherwise force a version of its own.

**Do not infer `i16` from `st16`.** `st16` is `storageBuffer16BitAccess`, a *storage* capability;
a conformant device may accept 16-bit data in a buffer and perform the arithmetic in f32. Rule V-15
in the namespace document forbids the inference, and the live test matches whole names for the same
reason — `contains("i16")` would be satisfied by the `st16` beside it in the same field.

Namespace document: `spec/namespaces/vulkan.md` at vocabulary version 5, rule V-15
([KISS#262](https://github.com/ThinkersJournal/KISS/pull/262)).

### Changed (breaking) — `Arith` widens to `u16`

`Arith(pub u8)` becomes `Arith(pub u16)`. Five bits were in use and the three new names fill a `u8`
exactly, so the next arithmetic name would have forced a breaking representation change on its own
account. Riding a break already being taken, for the same reason `ComponentType` became
`#[non_exhaustive]` during the version-2 cut.

### Added — the `vulkan:` namespace publishes a machine-readable vocabulary manifest

`kiss-vulkan-vocab/manifest/vulkan-vocabulary.json`, the KISS-CLASSIFY §6.8-0008 form of
`spec/namespaces/vulkan.md`, so a consumer binds against an artifact instead of hand-transcribing an
annex or hand-parsing its prose. `kind: "generated"`, emitted by
`examples/emit_vocabulary_manifest.rs` and byte-compared against the committed copy by
`tests/vocabulary_manifest.rs`.

The `vectors` array is the normative part, not `grammar`: two of the five fields choose their
encoding by a length-conditional switch, which no alphabet or regex expresses, so a consumer binding
only against the grammar can recognise every token and still produce the wrong one. Threshold
vectors sit at exactly 512 and 513 bytes for **both** length-conditional fields, which pins the
switch as `>` rather than `>=`.

New public API on `kiss-vulkan-vocab`: `VOCABULARY_VERSION`, `ComponentType::spelling`,
`OpClasses::alphabet`, `Arith::alphabet`, and `CoopMatrix`/`CoopVector::canonical_enumeration`.

### Changed (breaking) — the declared MSRV was never true; it is now measured

`vulkane` and `vulkan_gen` declared `rust-version = "1.85"` and **no consumer on 1.85 could
ever have compiled either of them**. Both now declare **1.88**, which is the floor that was
there all along. Two independent reasons, both predating the declaration:

- `vulkan_gen` uses **let-chains** (`if let ... && let ...`), stable only in 1.88. It is a
  *build-dependency* of `vulkane`, so this is on the consumer's compile path, not just ours.
- **`libloading 0.9` declares `rust-version = 1.88.0`** itself, so resolution fails before a
  single line is compiled.

The second is a property of the lockfile and the first is a property of our source, so the floor
does not move even for a consumer who resolves an older `libloading`.

`kiss-vulkan-vocab` and `vulkane_derive` **stay at 1.85** — measured, not assumed. Neither pulls
in a Vulkan loader, and a per-crate floor is worth more to a consumer than one number for the
workspace.

A new `MSRV` CI job builds every publishable crate at its own declared floor, **reading the
version out of `cargo metadata` rather than restating it in the workflow**. A hardcoded `1.88`
in CI would be a second copy of the promise, and the two drift the moment somebody edits the
manifest — which is precisely how this claim survived unexamined for as long as it did. The job
**builds and does not test**: a consumer compiles this crate, they do not run its dev-tests, and
gating on dev-dependencies would drag the floor up for reasons unrelated to the promise.

### Changed — the build toolchain is pinned, and it is not the same claim as the MSRV

`rust-toolchain.toml` pins **1.98.0** with `rustfmt` and `clippy`. This is the toolchain the
repo is *verified with*; the per-crate `rust-version` is the floor a *consumer* may compile
with. Neither number implies the other and CI now exercises both.

The pin exists because `stable` is a moving target resolved on different days on different
machines. A local `+stable` at 1.97.1 certified a branch that CI's `@stable` at 1.98.0 then
failed, on a lint that does not exist in 0.1.97 — so a green local run was not evidence about
CI at all.

`components` is not optional. Components are installed per toolchain *name*, so a bare
`channel` pin gets a toolchain without `rustfmt` or `clippy` and reds both jobs.

The MSRV jobs deliberately escape the pin: they pass an explicit `+<version>`, rustup's
highest-precedence selector. Verified rather than assumed — a pin that captured them would
have tested 1.98 under four leg names saying 1.85 and 1.88, which is a vacuous pass wearing
four green ticks. The existing toolchain-confirmation step is what would catch it.

### Fixed — `LICENSE-APACHE` was not the Apache License 2.0

All five copies diverged from the canonical text in two substantive places: §6 dropped "reasonable
and customary use in" from the trademark carve-out, and §9 substituted "Support" for "Additional
Liability" in both the section title and its operative clause — stating a narrower scope than the
licence it named. Replaced from `apache.org` and verified per-file against
`sha256 cfc7749b96f63bd3…`, including the blobs git stores rather than only the working tree.

**vulkane 0.13.0, kiss-vulkan-vocab 0.3.0 and vulkan_gen 0.5.0 shipped the wrong text and cannot be
corrected** — registry versions are immutable.

### Fixed — generated artifacts are pinned to LF

`.gitattributes` covered `LICENSE-*` only, so on a Windows checkout the vocabulary manifest and the
device fixture were stored as LF and checked out as CRLF. The committed manifest was therefore **not
byte-identical to a fresh emission**, and the emit-and-`git diff --exit-code` freshness gate
§6.8-0011 asks for could not be armed at all. The freshness test had been normalizing CRLF before
comparing, so it passed against a string neither party had on disk.

### Changed — versions

`kiss-vulkan-vocab` 0.3.0 → **0.4.0** and `vulkane` 0.13.0 → **0.14.0**, both breaking.

**0.4.0 rather than 0.3.x is the load-bearing part.** Vocabulary version 5 changes token bytes, and
under Cargo's 0.x rules `0.3.1` is *compatible* with the published `0.3.0` — so a consumer
requesting `kiss-vulkan-vocab = "0.3"` would have received version-5 tokens from what looks like a
patch upgrade. A breaking change wearing a compatible version number is exactly the silent kind:
nothing fails, the tokens simply stop matching the ones already in a cache.

## [0.13.0] — 2026-08-15

**First release since 0.10.1.** Versions 0.10.2, 0.11.0 and 0.12.0 were prepared and their entries
are recorded below, but they were **never published to crates.io** — the `vulkan:` vocabulary work
they carried was deliberately held until the token set could be cross-verified against a published
KISS artifact rather than against our own tests. 0.13.0 therefore rolls all of it up, and everything
under 0.10.2 and older in this file is included here.

**Cross-verified against KISS `origin/main` commit `b27e858` (2026-08-15).** At that commit
`spec/namespaces/vulkan.md` is *Vocabulary version 4* with clause V-1 requiring exactly five
`.`-separated fields, and `conformance/corpus/structure_key_vectors.json` carries
`namespace_vocabulary_versions = {"cuda": 1, "vulkan": 4}` together with the token
`vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none`. `kiss-vulkan-vocab`'s
`the_pinned_vector_matches_the_kiss_artifact_when_reachable` parses and re-spells that token
byte-exactly, reading it via `git show origin/main:…` — the *published* ref rather than a local
working tree, which could otherwise pass against an edit nobody pushed.

**This release run set `KISS_REQUIRE_PUBLISHED_REF=1`**, which turns the working-tree fallback into
a failure. The helper does still fall back when the published ref cannot be read — a contributor
whose KISS checkout has no remote is not doing anything wrong — but it now announces the degraded
mode and reports which source it compared against, and the release path refuses it outright.

That distinction is not hypothetical, and it was live while this release was being cut: the KISS
checkout on this machine sat at `b82b50b` with a working tree still carrying the **four-field
version-3 token**, while `origin/main` had moved to the five-field version-4 one. A silent fallback
would have verified this release against a stale tree. The corpus's own `coverage_note` records the
same failure in the other direction — a `vulkan:` suffix that "once byte-matched green while it was
MALFORMED against vulkan v4".

That commit is named here because a registry version is immutable and the question a reader asks
later is not "was this correct" but **"correct against what?"**. A provenance field that names the
wrong thing is unfalsifiable by anything that reads it, so this one names a commit that can be
checked. It is the only vector in the crate whose expected value has an author other than us;
every other test compares the crate to itself.

### Changed (breaking) — `vulkan:` vocabulary version 4 adds a fifth field and names the packed types

The token grows a fifth field, `<coopvec>`, spelled `cv-…`:

```
vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none
```

**Every four-field token is now invalid.** This is not a widening — `VulkanTarget::parse` requires
exactly five fields, and a version-3 token fails to parse rather than defaulting its new field.
Under §6.8-0002 byte-exact matching there is no degraded answer available: a token with the wrong
arity names nothing, so rejecting it is the honest outcome and a silently-defaulted `cv-none` would
have been a wrong cell rather than a partial one. Caches keyed on four-field tokens are stale and
must be **invalidated, not migrated**.

The new field enumerates the device's cooperative-*vector* combinations — five component types plus
a transpose flag per combination — canonically ordered, with the same length-triggered
`fnv1a64-<hex16>` digest escape the `<coop>` field uses above 512 bytes. `PhysicalDevice` gains
`cooperative_vector_properties()`, and `DeviceCapabilities` gains `coopvec`.

`VK_COMPONENT_TYPE_SINT8_PACKED_NV` and `VK_COMPONENT_TYPE_UINT8_PACKED_NV` are named `i8packed`
and `u8packed`.
`ComponentType` gains `S8Packed` and `U8Packed`, and `kiss::component()` maps both — a device
reporting them previously derived `x1000491000` / `x1000491001`.

Both changes ride one version bump deliberately. The packed enumerants are assigned at
`VK_HEADER_VERSION` 348, the registry baseline recorded for version 3, so under the namespace's §4
additive test naming them requires a bump on its own; folding them in with the fifth field costs one
version instead of two. **Version 4's own baseline is 348** — that is the number the next bump diffs
against.

These types are reachable *only* through the cooperative-vector query. Cooperative-matrix properties
never report them, which is why the query had to exist before the names could be derived at all,
and why this was verified against hardware rather than argued: an RTX 4070 reports both values in
`inputInterpretation` while its cooperative-matrix properties contain neither.

### Changed (breaking) — `vulkan:` vocabulary version 3 names FP8

`VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT` and `VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT` now derive
`f8e4m3fn` and `f8e5m2` instead of falling through to the `x<n>` escape. `ComponentType`
gains `F8E4M3FN` and `F8E5M2`, and `vulkane::kiss::component` maps both.

**This changes the bytes of tokens already in the wild.** A device reporting FP8
cooperative-matrix shapes previously derived `x1000491002` / `x1000491003`; it now derives
the named spellings. §6.8-0002 matching is byte-exact, so an old token and its version-3
equivalent **do not match**: caches keyed on version-2 tokens for such a device are stale
and must be **invalidated, not migrated**.

That this is a version bump rather than an additive change is not a judgement call. The
registered namespace's §4 additive test says to compare against the registry baseline
recorded for the previous version — `VK_HEADER_VERSION` 348 — and both enumerants are
already assigned at 348 (value `1000491002`, extension 492's block). Assigned at the
baseline means a conformant device could already have reported it, so a derivable token
could already have been affected by the absence of the name. **Bump.** Devices with no FP8
support are unaffected: their tokens never contained these values.

The `fn` suffix is mandatory and load-bearing. `f8e4m3fn` is the OCP OFP8 finite variant;
`f8e4m3fnuz` is a different format that KISS **reserves with no computation semantics**, so
a deriver must never emit it. `kiss-vulkan-vocab` now asserts both directions — no named
component spells a reserved dtype, and a reserved spelling does not parse as a named
component — because a `fnuz` slip is four characters that every round-trip test in the
crate would otherwise sail past.

`VK_COMPONENT_TYPE_FLOAT_E4M3_NV` / `..._E5M2_NV` are registry *aliases* of the EXT
enumerants, so one mapping covers both; a test pins that, since a future registry splitting
them would silently return NV-only drivers to `x<n>`.

Namespace document: `spec/namespaces/vulkan.md` at vocabulary version 3, rules V-10 and V-11.

### Changed (breaking) — `vulkan:` tokens spell signed integers with an `i` prefix

Part of the KISS `sk4` coordinated schema event. The registered `vulkan:` namespace was amended to
**vocabulary version 2** ([KISS#130](https://github.com/ThinkersJournal/KISS/pull/130)), taking
signed-integer component types from `s8`/`s16`/`s32`/`s64` to `i8`/`i16`/`i32`/`i64` to match the
`i` prefix used by KISS-Classify §6.1's `structure_key` dtype set. `kiss-vulkan-vocab` follows it
here.

**Every token naming a signed-integer component type changes bytes.** §6.8-0002 matching is
byte-exact and forbids subset or implication logic, so a version-1 token and its version-2
equivalent **do not match**. Any cache keyed on a version-1 token is stale and must be
**invalidated, not migrated** — a lookup with an old key will simply miss, and a store under an old
key will never be found again. Token length is unchanged, since each name keeps its width.

The document was amended **before** the crate, and this crate's
`tests/registered_namespace.rs` guard moved in the same commit as the `ComponentType` arms.

**Variant names are unchanged and deliberately do not match their tokens.** `ComponentType::S8`
now spells `i8`. The variant is named for its Vulkan source (`VK_COMPONENT_TYPE_SINT8_KHR`); the
token follows the KISS-facing wire vocabulary. Do not "fix" either to match the other.

### Changed (breaking) — `ComponentType` is `#[non_exhaustive]`

Enum hardening, riding this breaking cut so that future dtype additions do not force another one
(KISS `sk4` RFC §10). Downstream `match` expressions over `ComponentType` now require a wildcard
arm.

**Migration note, required by RFC §10 — this trades a build break for a silent behaviour change.**
Because new variants can now land without a major version, a value that previously arrived as
`ComponentType::Other(n)` and matched a caller's catch-all will, once it is named, match the new
variant instead — with **no compile error to announce it**. If you branch on `Other(n)` for a
specific `n`, that branch can stop being taken by a dependency upgrade alone. This is the same
reclassification hazard as the `bf16` fix below, made permanent as a property of the type.

### Changed — versions

`kiss-vulkan-vocab` 0.1.0 → **0.3.0**, `vulkan_gen` 0.3.0 → **0.5.0**, and `vulkane` 0.10.1 →
**0.13.0**. All breaking, and all published together: the intermediate bumps recorded during this
work (`kiss-vulkan-vocab` 0.2.0, `vulkane` 0.11.0 / 0.12.0) were never released, so the
crates.io jump is straight from the last published versions.

The FP8 variants noted as outstanding here during development are **shipped** — see the version-3
entry above. They were **resolved by exclusion**, not by the authored §6.1 layout⇒variant table
that was originally expected to unblock them: KISS reserves the `fnuz` spellings with no
computation semantics, which leaves `fn` as the only spelling either Vulkan enumerant can denote.
Adding them was *additive* to the Rust type rather than breaking, thanks to `#[non_exhaustive]`
above — which is the payoff that change was made for. The *token* bytes still changed, which is
what drove the vocabulary version bump.

### Fixed (soundness) — driver-written enum fields could hold an invalid discriminant

Reading a Rust `enum` whose memory holds a discriminant outside its declared set is **undefined
behaviour**. Every struct that `vk.xml` marks `returnedonly="true"` is filled by the implementation, and
an implementation may report a value this `vk.xml` has never heard of — a component type or driver
ID from an extension newer than the pinned spec. The UB was therefore reachable by **upgrading a
graphics driver**, with no application change and no error path to observe.

The generator already emitted a checked `from_raw(i32) -> Option<Self>` on every enum. Typing the
field as the enum guaranteed it could never be called: the check existed and enforced nothing.

**Driver-written enum fields are now emitted as raw `i32`** — 81 fields across the `returnedonly`
structs — so the conversion becomes an explicit decision. Two deliberate exclusions: `sType` keeps
its enum type (the application writes it even in a `returnedonly` struct, which is how a `pNext`
query chain is assembled), and pointer/array fields are untouched (reading through a raw pointer is
already `unsafe` and carries its own contract).

Application-written structs are unaffected. Roughly 1,500 enum-typed fields keep ergonomic enums
and exhaustive matching, because an application only ever writes values it obtained from the enum.

**Breaking.** Safe-layer accessors that read these fields now return `Option<T>`, each paired with a
`_raw` accessor:

- `CooperativeMatrixProperties::{a,b,c,result}_type()` and `scope()` → `Option<_>`, plus
  `*_raw() -> i32`
- `PhysicalDeviceProperties::device_type()` → `Option<PhysicalDeviceType>`, plus `device_type_raw()`
- `SurfaceFormat::format()` / `color_space()` → `Option<_>`, plus `format_raw()` / `color_space_raw()`
- `DriverProperties::driver_id` → `Option<VkDriverId>`, and the new field `driver_id_raw: i32`

The pairing is load-bearing, not convenience. The checked form is **lossy by construction**: every
unrecognized value collapses to `None`, so two different unknown driver IDs become indistinguishable
through it, and a shader cache keyed on that would reuse one driver's binaries for another. Key
caches on the raw value; gate workarounds on the named one.

`Swapchain::pick_surface_format` now skips formats this build cannot name, and returns
`ERROR_FORMAT_NOT_SUPPORTED` if none remain. Use the `_raw` accessors and build the swapchain
directly if you need one of those formats.

### Fixed — `bfloat16` was spellable but not derivable from a device

`kiss::component()` maps a raw `VkComponentTypeKHR` to the KISS vocabulary's `ComponentType`.
It covered the base set (values 0–10) but had no arm for `VK_COMPONENT_TYPE_BFLOAT16_KHR`, so a
driver reporting bfloat16 cooperative matrices yielded `ComponentType::Other(1000141000)`.

`ComponentType::BF16` and its `bf16` token have always existed and round-trip correctly, which is
why the vocabulary crate's tests passed throughout — they construct the variant directly. Nothing
exercised the derivation from a live device, so the gap sat between two things that were each
individually fine.

**This is an `Other(n)` → named reclassification.** Downstream code matching `ComponentType` with
a wildcard arm compiles unchanged and behaves differently: a bfloat16 cooperative-matrix shape
that previously fell into a catch-all now matches `BF16`. Tokens derived from such a device change
accordingly — which is the point, but it does mean a cache keyed on the old token is stale.

When that fix landed, four further values were left unmapped deliberately. **All four are now
named**, by the version-3 and version-4 entries above — this paragraph records why they were held,
because the reasoning is what the two bumps had to answer:

- `FLOAT8_E4M3_EXT` / `FLOAT8_E5M2_EXT` were blocked on the KISS `sk4` schema event: the names
  denote a *layout*, KISS §3.1.5 makes the `fn`/`fnuz` suffix mandatory, and the Vulkan registry
  never says which variant is meant. **Resolved by exclusion in version 3** — KISS reserves the
  `fnuz` spellings with no computation semantics, so `fn` is the only spelling either enumerant
  can denote.
- `SINT8_PACKED_NV` / `UINT8_PACKED_NV` were held because they carry `s8`/`u8` data in a packed
  layout — a different shader-side contract from the unpacked types — and folding them onto
  `S8`/`U8` would have collapsed two distinct Vulkan values onto one token, letting a packed-only
  device satisfy a target asking for plain `s8`. **Resolved in version 4 by giving them their own
  spellings** (`i8packed` / `u8packed`) rather than an alias, which keeps the two contracts
  distinct while making the values derivable.

`ComponentType` is now `#[non_exhaustive]` (see above), so naming a value no longer requires a
coordinated major on its own account — the version bump is driven by the token bytes changing, not
by the Rust type.

## [0.10.2] — 2026-08-07

Found by a sweep for unfinished work prompted by Fuel and Baracuda each turning up
undocumented TODOs, one of which had caused several bugs before anyone spotted it. Grepping
`TODO` in this repo is a false all-clear — the deferred work here is written as prose
(`// … for now`), and that is where the vertex-format bug below was hiding.

### Fixed — `#[derive(Vertex)]` gave signed-integer fields a float format

`[i32; 3]` mapped to `Format::R32G32B32_SFLOAT` because no `R32G32B32_SINT` constant existed
when the derive was written. Both formats are 12 bytes with identical vertex-buffer layout, so
this compiled, passed the validation layers, and produced no error at any point — the GPU simply
reinterpreted each integer's bit pattern as IEEE-754. A vertex attribute of `[-1, 0, 1]` reaches
the shader as roughly `[-1e-45, 0.0, 1.4e-45]`.

- `[i32; 3]` now maps to `R32G32B32_SINT`.
- `[i32; 4]` is now supported, mapping to `R32G32B32A32_SINT`. It previously failed to compile.
- `Format::R32G32B32_SINT` and `Format::R32G32B32A32_SINT` are new public constants, and both
  are present in `Format::bytes_per_pixel` (12 and 16). The size lookup had the same gap, so
  adding the constants alone would have moved the defect rather than closed it.

Anyone deriving `Vertex` on a struct with an `[i32; 3]` field was reading garbage in the shader;
after upgrading, that attribute delivers the integers it always should have. Shaders written to
compensate for the old behaviour by declaring the input as `vec3` will need to change to `ivec3`.

### Fixed — the bindings generator emitted its fragments in a nondeterministic order

`CodeAssembler::resolve_dependencies` seeded its topological sort by iterating a `HashSet`, whose
order varies per process. Consecutive builds of the same `vk.xml` therefore produced different
`vulkan_bindings.rs` files.

Item order carries no meaning in Rust, and the emitted API was verified identical across three
differently-ordered builds — same 3,353 type definitions, same 337 deduplication decisions — so
no incorrect bindings were ever generated. The reason it still mattered: the deduplication pass
keeps whichever definition it encounters *first* and strips later ones, so the surviving
definition of any colliding pair was a function of that nondeterministic order. Generation is now
reproducible, verified byte-identical across builds.

### Changed — generator internals (`vulkan_gen` 0.4.0, breaking)

- Deduplication now determines what a fragment defines by scanning the emitted code rather than
  trusting a module's self-reported `defined_types`. Those claims are partly hand-maintained; a
  name listed but not emitted marked the type as seen and would have stripped a later fragment's
  real definition. One such phantom entry was live, and its only visible effect was a spurious
  "skipped a definition here" comment in the generated file.
- **Removed `CodeAssembler::validate_generated_code` and `AssemblerError::DuplicateType`.** The
  function was never called and could not be: fragments legitimately redefine names that earlier
  fragments emitted, which is what the deduplication pass exists to absorb — 337 times in a full
  run — so enabling it as an error would have failed the build on the normal case. Duplicate and
  undefined types are already checked by `type_integration::check_data_consistency`, which runs
  earlier over the intermediate JSON and does fail the build.
- `GeneratorMetadata` is now documented as informational. Nothing consults `priority` (ordering
  comes from the dependency graph) and nothing validates against `defined_types` / `used_types`.
- **`SafeHandlesStats` gained an `unclassified` field.** The auto-RAII generator skips handles
  that don't fit its create/destroy shape, and a skip was invisible: the build succeeded and the
  handle simply had no safe wrapper. Every handle is now sorted into hand-written, auto-wrapped,
  or explicitly excluded with a recorded reason (`KNOWN_UNWRAPPABLE`); anything else is reported
  here and logged as a build warning. Against the pinned `vk.xml` the 59 non-alias handle types
  are 30 + 25 + 4 with nothing left over, and a test enforces that rather than asserting it in a
  comment. The build stays permissive so a newer `vk.xml` still works.
- Unsuffixed hexadecimal constants are typed by magnitude instead of assumed `u32`. A literal
  wider than 32 bits previously emitted a `u32` constant that the compiler rejected as out of
  range.

### Fixed — documentation

- `CONTRIBUTING.md` was stale from the `infra-vulkan` → Vulkane rename: wrong crate name, wrong
  clone URL, Rust 1.75 against an edition-2024 crate, a `--features validation-layers` command
  for a feature that does not exist, and a doc example calling `device.allocate_memory(…)`, which
  is not a method. CMake is required by the `shaderc` and `slang` features, not by the examples.
- Removed a `VulkanTag::is_deprecated` stub from the generated tag registry that always returned
  `false`. `vk.xml` `<tag>` elements carry only `name`, `author` and `contact` — deprecation is a
  property of extensions, so the question has no answer rather than an unimplemented one.

## [0.10.1] — 2026-08-01

### Fixed — allocator memory-type selection and failure-path cleanup

Both issues surfaced from [Fuel](https://github.com/ciresnave/fuel)'s audit of a machine-level
`VIDEO_MEMORY_MANAGEMENT_INTERNAL` bugcheck (subcode `0x2D`, `DdiMapCpuHostAperture failed`) on
2026-07-31 — CPU host aperture / PCIe BAR exhaustion with ReBAR enabled. Neither is proven to
have caused that crash, but both make host-mapping pressure worse rather than better.

- **`AllocationUsage::HostVisible` no longer resolves to BAR memory when ordinary host-visible
  memory exists.** `pick_memory_type` passed identical `required` and `preferred` flag masks for
  this usage, which made `find_type`'s preference pass identical to its fallback pass — so
  selection degenerated to "first matching type wins" and the documented distinction between
  `HostVisible` and `HostVisibleDeviceLocal` was never enforced. Staging buffers could therefore
  land in the PCIe BAR window, whose exhaustion surfaces as a driver-level failure rather than a
  catchable `VK_ERROR_OUT_OF_DEVICE_MEMORY`.

  Selection now *avoids* `DEVICE_LOCAL` for `HostVisible` as a preference, falling back when no
  other host-visible type exists (UMA and fully-host-visible-VRAM layouts still work). Note this
  is narrower than it may sound: the Vulkan spec already requires a memory type whose
  `propertyFlags` are a strict subset of another's to be enumerated first, so on most layouts the
  plain host type was already winning. The exposure is layouts where the flag sets are
  *incomparable* — `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` versus
  `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, neither a subset of the other — because there the
  spec imposes no ordering and a conforming driver may enumerate the BAR type first.

- **`vkDeviceMemory` is no longer leaked when `vkMapMemory` or sub-allocation fails.** Three
  allocation paths (default pool growth, custom pool growth, dedicated allocation) allocated a
  block and then propagated a later failure with `?`, stranding the handle: nothing owned it, so
  neither `Drop` nor `destroy_pool` could reclaim it. Blocks are 64 MiB, or 256 MiB on heaps
  ≥ 4 GiB. The reachable trigger is the damaging one — `VK_ERROR_MEMORY_MAP_FAILED` is what a
  driver reports when it cannot satisfy a host mapping, so the leak sat on the recovery path and
  made each retry likelier to fail than the last. Two of those paths could also strand a *live
  mapping*, holding host aperture rather than merely heap space.

Selection is covered by unit tests over synthetic memory layouts (`pick_memory_type_in`), including
the incomparable-flag-set case that no single machine exhibits. The leak fix is structural and
verified by inspection — forcing `vkMapMemory` to fail would require a mock ICD.

## [0.10.0] — 2026-07-31

Additive: a new optional feature and one new query. Nothing existing changes behaviour.

The `vulkan:` namespace of KISS-Classify §6.8 was registered to this crate on 2026-07-31
([KISS #120](https://github.com/ThinkersJournal/KISS/pull/120)), with the vocabulary pinned in
`spec/namespaces/vulkan.md` there. This release ships the two halves that implement it.

### Added — KISS `vulkan:` target_capability derivation (`kiss-target` feature)

- **New sibling crate [`kiss-vulkan-vocab`](kiss-vulkan-vocab/)** — the `vulkan:` capability-set vocabulary for KISS-Classify §6.8 `target_capability` tokens: canonical spelling, parsing, and byte-exact comparison. **Zero dependencies and no Vulkan linkage**, because KISS-CLASSIFY-6.9-0003 forbids producing or parsing a token from loading a compute driver — a conformance implementation must manage with its standard library alone. That constraint is enforced structurally by [`tests/zero_dependency.rs`](kiss-vulkan-vocab/tests/zero_dependency.rs), which reads the manifest and fails the build if a dependency table is ever non-empty, rather than by a comment nobody checks.

  Token grammar: `vulkan:<subgroup>.<ops>.<arith>.<coop>` — four fixed-position fields separated by `.`, parts within a field by `-`. Every set is canonically sorted and every field always present, so two independent implementations spelling the same target produce identical bytes, which §6.8-0002 requires (it permits no matching tolerance whatsoever). Legal-but-non-canonical spellings are **rejected rather than normalized**: under byte-exact matching, accepting two spellings of one target would let them silently fail to match each other. Long cooperative-matrix shape lists fall back to an `fnv1a64-<hex>` digest, triggered strictly by the encoded length of the canonical enumeration (`COOP_DIGEST_THRESHOLD`, 512 bytes — the same string the digest hashes, so there is nothing to define twice) and never by implementation preference — a preference-driven switch would let two honest derivers emit different tokens for the same target.

- **`PhysicalDevice::shader_arithmetic_features() -> Option<ShaderArithmeticFeatures>`** — `shaderFloat16` / `shaderInt8` (Vulkan 1.2 core) plus the 16-/8-bit storage-buffer access features, via `vkGetPhysicalDeviceFeatures2`. These gate whether a half-precision or quantized kernel can exist on a device at all, making them a specialization axis rather than a tuning knob. Compute precision and *storage* precision are reported separately — a device may accept 16-bit data in a storage buffer while doing the arithmetic in f32. Gated on `effective_api_version()` with the same honest-`None` discipline as the other property queries. New public type `safe::ShaderArithmeticFeatures`.

- **`vulkane::kiss` module** (behind the new optional `kiss-target` feature, off by default) — derives `vulkan:` tokens from a live `VkPhysicalDevice`. The API is deliberately **not** `device -> token`: a `target_capability` names the specialization a kernel was *built for*, not the device's capability envelope, so on a device with a 32..=64 pinnable range a wave32-pinned and a wave64-pinned kernel are different cells and an envelope-shaped token would collide them. Instead `DeviceCapabilities::of()` reads the envelope, `admissible_subgroups()` enumerates the choice axis, `target_for()` spells one concrete token per choice, and `admits()` answers the capability question — kept separate from token matching, since §6.8-0002 forbids a consumer from applying subset or implication logic when matching.

  Verified on live hardware: an AMD Radeon 610M yields three distinct tokens (`sgdyn`, `sg32`, `sg64`) from one device — the "a device admits a *set* of tokens" property demonstrated rather than asserted — with derivation proven deterministic across repeated reads, since driver-reported cooperative-matrix order is not guaranteed stable and an unsorted list would produce a token that differs run to run.

## [0.9.0] — 2026-07-30

Minor bump rather than a patch because `cooperative_matrix_properties` loses its `unsafe`
qualifier. The change only *loosens* a caller's obligations and is semver-compatible, but it
alters a public function signature, and callers who wrapped the call in `unsafe { .. }` will
newly see `unused_unsafe` — which fails a build using `-D warnings`. See the migration note
below.

### Added — compilation-target capability queries

These three fill the inputs a caller needs to describe *what a Vulkan device specializes a compute kernel for*. They are the Vulkan-side raw material for the `vulkan:` namespace of the KISS-Classify §6.8 `target_capability` descriptor, whose per-namespace vocabulary is owned by the namespace maintainer.

- `PhysicalDevice::subgroup_properties() -> Option<SubgroupProperties>` — subgroup ("wave"/"warp") width, supported stages, supported operation classes, and quad-in-all-stages, from `VkPhysicalDeviceSubgroupProperties` (Vulkan 1.1 core); plus `size_control: Option<SubgroupSizeControl>` (min/max pinnable subgroup size, max compute-workgroup subgroups, and which stages accept a pinned size) from `VkPhysicalDeviceSubgroupSizeControlProperties` (Vulkan 1.3 core / `VK_EXT_subgroup_size_control`). **This closes a hole Vulkane documented against itself**: [`ComputePipelineOptions::required_subgroup_size`](vulkane/src/safe/pipeline.rs) has always let a caller *pin* a subgroup size, and its own doc comment says the value must lie within `minSubgroupSize..=maxSubgroupSize` — bounds Vulkane offered no way to read. `SubgroupSizeControl::permits` / `permits_in_compute` validate a candidate size (power-of-two, in range, stage accepts a pinned size) before pipeline creation. **Note the instance-version prerequisite:** size control is Vulkan 1.3 core (or `VK_EXT_subgroup_size_control`), and the gate is on `effective_api_version()` — so `size_control` is `None` on an `Instance` created below 1.3 *regardless of what the device supports*, and `InstanceCreateInfo::api_version` defaults to `V1_0`. A caller at 1.2 sees an honest `None` and must raise the instance version to pin a subgroup size. The base `subgroup_properties()` query needs only 1.1. Subgroup width is the single most important Vulkan kernel-specialization axis: 32 on NVIDIA, 64 on AMD GCN/CDNA, *either* on RDNA, 8/16/32 on Intel. New public types `safe::SubgroupProperties`, `safe::SubgroupSizeControl`, `safe::SubgroupFeatureFlags`.
- `PhysicalDevice::driver_properties() -> Option<DriverProperties>` — `driver_id` (the `VkDriverId` naming the ICD), `driver_name`, `driver_info`, and the claimed CTS `conformance_version`, from `VkPhysicalDeviceDriverProperties` (Vulkan 1.2 core / `VK_KHR_driver_properties`). `PhysicalDeviceProperties::driver_version` is a bare `u32` whose bit-packing is *vendor-defined* (NVIDIA packs it (22,14,6,10), AMD (22,10,10,10), Intel-on-Windows (18,14)), so it cannot be decoded portably and is good only for equality. This gives a portable, legible driver identity for shader-cache keys and a proper enum for driver-quirk gating — RADV and AMDVLK drive the same hardware and do not make the same codegen choices. New public types `safe::DriverProperties`, `safe::ConformanceVersion`.
- `PhysicalDevice::effective_api_version() -> ApiVersion` — `min(instance apiVersion, device apiVersion)`, the version that actually governs property queries. A Vulkan implementation must behave as the version the *instance* requested, so an instance created at 1.0 leaves 1.1+ `pNext` property structs untouched **even on a 1.3 device**, and a caller gating on the device version alone reads a zeroed struct back as though it were an answer. Both queries above gate on this and return an honest `None` instead. Note `InstanceCreateInfo::api_version` defaults to `V1_0`, so raise it if a query declines unexpectedly.

### Changed — `cooperative_matrix_properties` is now safe

- `PhysicalDevice::cooperative_matrix_properties()` is no longer an `unsafe fn`. It previously placed the burden on the caller to have enabled `VK_KHR_cooperative_matrix`, because the Vulkan loader hands back a non-null stub for any KHR entry point whose *name* it knows and that stub can crash on a device that doesn't implement the extension (notably Mesa Lavapipe). The method now asks the device itself via `enumerate_extension_properties()` and returns an empty `Vec` when the extension isn't advertised, so the stub is never reached — the same honest-gating discipline `device_identity()` applies to `VK_EXT_pci_bus_info`. An empty `Vec` is therefore unambiguous rather than possibly-undefined.

  **Migration:** remove the `unsafe { .. }` wrapper — leaving it compiles but raises `unused_unsafe`, which bites under `-D warnings`. The redundant extension check *immediately guarding this call* can go too, but **keep the flag if it also gates anything else**: a caller who checked `VK_KHR_cooperative_matrix` once and reused that boolean for `DeviceFeatures::with_cooperative_matrix()`, `DeviceExtensions::khr_cooperative_matrix()`, or pipeline construction must retain it. Deleting the binding outright would leave the extension never enabled at device creation while this self-gating query still reports shapes from the *supported* list — a silently disabled cooperative-matrix path rather than a build error. This change also made the API runtime-testable for the first time — `safe_wrapper_test.rs` previously carried a standing note that no test could exist because calling the function on CI's Lavapipe was undefined behaviour; that note is replaced by a real test.

## [0.8.3] — 2026-06-28

### Added — physical-device identity

- `PhysicalDevice::device_identity() -> Option<DeviceIdentity>` exposes the device's stable identity for out-of-band correlation: `device_uuid` / `driver_uuid` (always, from `VkPhysicalDeviceIDProperties`, Vulkan 1.1 core), `device_luid` (`Some` only when the platform marks it valid — Windows) plus its `device_node_mask`, and `pci: Option<PciBusInfo>` (`Some` only when the device advertises `VK_EXT_pci_bus_info`). One `vkGetPhysicalDeviceProperties2` call, gated honestly: `None` when props2 is unavailable, and each sub-field is `Some` only when its source is actually present. This is the *join key* a caller needs to match a `VkPhysicalDevice` against an out-of-band GPU source — NVML by UUID, DXGI/D3DKMT by LUID, Linux sysfs (`gpu_busy_percent`) by PCI address — or against the same device seen through CUDA/D3D/OpenGL. Added because Vulkan exposes **no** cross-process GPU load / utilization / queue-depth query beyond the VRAM `memory_budget`; identity is the most Vulkane can (and should) provide toward that, with the load lookup itself living in a separate, API-agnostic layer. New public types `safe::DeviceIdentity` and `safe::PciBusInfo`.

### Added — Profile v1 conformance lock-in

- Vulkane is confirmed conformant to Fuel's **Kernel-Seam Interop Contract — Profile v1** (ratified 2026-06-20) in its **FDX-only, BDA-subset** role. No API change was required — the contract pins Vulkane to a *named surface*, all of which shipped in 0.8.2: `AllocatorOptions::buffer_device_address` / `Allocator::new_with_options`, `BufferUsage::SHADER_DEVICE_ADDRESS`, `DeviceFeatures::with_buffer_device_address`, and `Buffer::device_address`. Added [`tests/profile_v1_conformance.rs`](vulkane/tests/profile_v1_conformance.rs), a compile-time lock-in (mirroring the `Send + Sync` lock-ins on `Queue` / `CommandBuffer`) so a future rename, removal, or signature change of any named-surface item fails Vulkane's CI rather than `fuel-vulkan-backend`'s build. This operationalizes the contract's §7.2 rule that *a Vulkane major bump triggers a re-check of the named surface* — the surface is pinned by behavior, not by a `>= 0.8.2` version floor.

## [0.8.2] — 2026-06-19

### Added — device-address-capable allocator

- `Allocator::new_with_options(device, physical, AllocatorOptions { buffer_device_address: true })` makes every `VkDeviceMemory` block the allocator allocates carry `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`. Buffers sub-allocated from such an allocator (created with `BufferUsage::SHADER_DEVICE_ADDRESS`) now return a valid GPU virtual address from `Buffer::device_address()`. Previously the flag was only set on the manual `DeviceMemory::allocate_with` path, so addresses read from pooled or `Buffer::new_bound` buffers were invalid on strict drivers. The flag lives on the block (not the buffer) because one block backs many sub-allocations, mirroring VMA's `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT`. Requires the `bufferDeviceAddress` device feature. `Allocator::new` is unchanged (defaults to `buffer_device_address: false`). Unblocks downstream consumers (Fuel) that address tensors via `buffer_reference` in shaders, where the tensor's data pointer is a `VkDeviceAddress`.

## [0.8.1] — 2026-05-21

### Added — thread-safety markers

- `safe::Queue` is now `Send + Sync` via `unsafe impl`. Vulkan queues have no thread affinity at the API level, but the application owns external synchronization per `VkQueue` for `vkQueueSubmit` / `vkQueueWaitIdle` / `vkQueueBindSparse` / `vkQueuePresentKHR`. Callers can share `&Queue` across threads (e.g. via `Arc<Mutex<Queue>>` or a scheduler) so long as concurrent submissions on the same queue handle are prevented. Unblocks downstream consumers (Fuel) that build work on worker threads and submit through a serializer.
- `safe::CommandBuffer` is now `Send + Sync` via `unsafe impl`, for the same reason. Recording APIs take `&mut self`, so the Rust borrow checker already prevents concurrent recording on the same buffer; sharing `&CommandBuffer` across threads is sound. The per-pool external-sync contract for `vkFreeCommandBuffers` (called from `Drop`) remains the caller's responsibility.
- Compile-time `Send + Sync` lock-in assertions added for both types so future field additions cannot silently regress this guarantee.

## [0.8.0] — 2026-04-21

Coverage release: full ray-tracing surface, external-memory / external-semaphore interop, synchronization-2 barriers, push descriptors, dynamic rendering, descriptor-buffer binding, timeline semaphores, subgroup-size control, memory priority, and generator-emitted ergonomic safe signatures for ~545 Vulkan commands. Two latent codegen correctness bugs fixed along the way.

### Added — curated extension wrappers

- **Ray tracing** — `safe::AccelerationStructure` (BLAS/TLAS/Generic, AABB + Triangles + Instances geometry), `safe::RayTracingPipeline` with `ShaderGroup` enum (General/TrianglesHit/ProceduralHit) and `ShaderBindingRegion`, `PhysicalDevice::ray_tracing_pipeline_properties`, `CommandBufferRecording::build_acceleration_structure` / `bind_ray_tracing_pipeline` / `trace_rays`, `Device::acceleration_structure_build_sizes`. Examples: [`ray_tracing_as_build`](vulkane/examples/ray_tracing_as_build.rs) builds a live BLAS + TLAS on the local GPU end-to-end.
- **External memory / semaphore interop** — `DeviceMemory::get_win32_handle` / `get_fd`, `Semaphore::get_win32_handle` / `get_fd` / `import_win32_handle` / `import_fd`, `Win32Handle` newtype, `SemaphoreImportWin32` / `SemaphoreImportFd`. Unblocks CUDA, HIP, DX12, and DMA-BUF bridging. Example: [`external_memory_export`](vulkane/examples/external_memory_export.rs).
- **Synchronization 2** — `CommandBufferRecording::memory_barrier2` / `image_barrier2` / `buffer_barrier2` with 64-bit `PipelineStage2` / `AccessFlags2`.
- **Dynamic rendering** — `CommandBufferRecording::begin_rendering` / `end_rendering` with `RenderingInfo` + `RenderingAttachment`.
- **Push descriptors** — `CommandBufferRecording::push_descriptor_set` taking a `&[PushDescriptorWrite]` that hides the `VkWriteDescriptorSet` layout.
- **Descriptor buffer** (`VK_EXT_descriptor_buffer`) — `DescriptorSetLayout::descriptor_buffer_size` / `descriptor_buffer_binding_offset` queries, `CommandBufferRecording::bind_descriptor_buffers` / `set_descriptor_buffer_offsets`.
- **Timeline semaphores** — `Semaphore::timeline_with_pnext` composes caller-supplied chains with the mandatory `VkSemaphoreTypeCreateInfo`.
- **Compute pipeline options** — `ComputePipelineOptions` carries `required_subgroup_size` (`VK_EXT_subgroup_size_control`), `specialization`, and `cache` in one bag; `ComputePipeline::with_options` is the general constructor.
- **Memory priority** — `MemoryAllocateInfo::priority: Option<f32>` auto-chains `VkMemoryPriorityAllocateInfoEXT`.
- **Shader integer dot product** — `PhysicalDevice::shader_integer_dot_product_properties() -> ShaderIntegerDotProductProperties` with `has_any_int8_acceleration()` helper.
- **pNext extension points** on every safe create-info builder: `DeviceCreateInfo::pnext`, `InstanceCreateInfo::pnext`, `MemoryAllocateInfo::pnext`, plus new `with_pnext` constructors on `Buffer`, `Image`, `Fence`, and `Semaphore`. Any unwrapped extension struct can now be layered on without dropping to raw.

### Added — generated ergonomic traits (Phase 3)

- `DeviceSafeExt`, `InstanceSafeExt`, `PhysicalDeviceSafeExt`, `QueueSafeExt`, `CommandBufferRecordingSafeExt` — auto-generated per-command methods with idiomatic Rust signatures alongside the raw Phase-2 `DeviceExt` etc. traits. **545 ergonomic methods** emitted from `vk.xml`:
  - **Slice coalescing** — `(count: u32, const T*)` pairs collapse into `&[T]` inputs. `cmd_pipeline_barrier(..., &[MemoryBarrier], &[BufferMemoryBarrier], &[ImageMemoryBarrier])` is one signature.
  - **Enumerate** — `(*mut u32 count, *mut T data)` pairs become `Result<Vec<T>>` / `Vec<T>` return types. `enumerate_physical_devices` issues the classic two-call count-then-fill idiom automatically.
  - **Single-output** — trailing `*mut T` parameters become `Result<T>` returns (`get_memory_win32_handle_khr(info: &…) -> Result<HANDLE>`).
  - **Reference input structs** — `*const T` parameters become `&T`.
  - **Scalar return passthrough** — `VkDeviceAddress` / `VkBool32` / typed handles pass through untouched (`get_buffer_device_address(info: &…) -> VkDeviceAddress`).
  - Commands with unsupported shapes (pointer-to-pointer, parallel slices sharing one count, `len` pointing inside a struct) fall through to the raw Phase-2 traits — no method emitted.

### Fixed — generator correctness

- **Nested C-array layout** — `VkTransformMatrixKHR.matrix[3][4]` was emitted as `[f32; 3]` (12 bytes) instead of `[[f32; 4]; 3]` (48 bytes). Every multi-dimensional `float matrix[a][b]` field in `vk.xml` was silently truncated. Fixed in `struct_gen::map_type_to_rust`; any ray-tracing workload using `VkAccelerationStructureInstanceKHR` was affected.
- **Transitive extension-dep walker** — `transitive_requires` harvested per-`<require>` `depends` attributes and treated them as extension prerequisites. In `vk.xml` those attributes mark *conditional* enum inclusion (e.g. "expose these extra debug-report enums if the user also enables debug_report"), not dependencies. Enabling `VK_KHR_acceleration_structure` therefore silently tried to enable `VK_EXT_debug_report` (an *instance* extension) at device creation, causing `ERROR_EXTENSION_NOT_PRESENT` on every driver. Fixed to use only the canonical top-level `requires` attribute.

### Breaking

- `DeviceCreateInfo` gained a `pnext: Option<&PNextChain>` field (default `None`). Callers using `..Default::default()` are unaffected; anyone constructing the struct with explicit named fields must add it (or switch to update syntax).
- `InstanceCreateInfo` gained the same `pnext` field.
- `MemoryAllocateInfo` gained `pnext` and `priority` fields. Direct struct-literal callers must supply both or use `..Default::default()` — the struct now derives `Default`.
- `CommandBufferRecording::memory_barrier2` / `image_barrier2` / `buffer_barrier2` return `Result<()>` (Sync2 function pointers may be absent on pre-1.3 devices without `VK_KHR_synchronization2`). Previously no sync2 methods existed, so this is only new-code exposure.
- `ComputePipeline::with_specialization_and_cache` is retained as a shim; new callers should prefer `ComputePipeline::with_options`.

### Test + example coverage

- 249 total workspace tests pass. 10 new generator pattern-matcher unit tests. 5 new live-device tests exercising generated ergonomic traits against a real driver. 2 new example programs that **run live** against the local GPU — `external_memory_export` exports a real Win32 HANDLE, `ray_tracing_as_build` builds a real BLAS + TLAS on the RT hardware.

## [0.7.0] — 2026-04-19

Allocator-side VRAM observability: the `Allocator` can now surface the driver's per-heap budget numbers in one call, fire budget-pressure callbacks when usage crosses a configurable threshold, and predictively check whether a prospective allocation would exceed the budget — all without requiring the user to opt into `VK_EXT_memory_budget` manually.

### Added

- **`Allocator::vram_budget()` / `Allocator::vram_used()`** — scalar-byte convenience helpers that sum the driver-reported budget and usage across every `DEVICE_LOCAL` memory heap. The single-number answer ML schedulers, profilers, and UI indicators typically want.
- **`Allocator::has_memory_budget_support()`** — returns `true` iff the budget numbers are authoritative (both `vkGetPhysicalDeviceMemoryProperties2` is loaded *and* `VK_EXT_memory_budget` is enabled on the device). Use this to distinguish "heap is empty" from "no query support".
- **Budget-pressure callback registry** — `Allocator::register_pressure_callback(threshold, hysteresis, closure)` fires a `PressureEvent` when a heap's `usage / budget` fraction rises past `threshold` (`PressureKind::Crossed`), falls back below `threshold - hysteresis` (`PressureKind::Relieved`), or — via `would_fit` — is projected to rise past `threshold` on a pending allocation (`PressureKind::Predictive`). Per-heap hysteresis latching prevents flutter near the threshold. Callbacks are invoked after every internal allocator lock has been released, so they may call back into the `Allocator` without deadlocking. `unregister_pressure_callback(id)` removes a registration.
- **`Allocator::would_fit(size, memory_type_index) -> FitStatus`** — proactively computes whether a forthcoming allocation would keep usage under the driver's soft budget, fires `Predictive` events for any threshold it would cross, and returns the projected heap stats (`current_usage`, `budget`, `projected_usage`, `projected_fraction`, `fits`). Lets schedulers free resources *before* attempting an allocation rather than reacting to a `Crossed` event after the fact.
- **`Device::enabled_extensions()` / `Device::is_extension_enabled(name)`** — introspect the final extension list sent to `vkCreateDevice`. Captures both explicit user requests and any extension the safe wrapper auto-enabled.
- **New documentation**: `vulkane/docs/DEFRAG_FOR_ML.md` — a dedicated walkthrough of the existing `build_defragmentation_plan` / `apply_defragmentation_plan` API aimed at ML-framework integrators, including a full worked tensor-pool compaction example and guidance on layering defrag under budget-based eviction.

### Changed

- **Device creation now auto-enables `VK_EXT_memory_budget`** when the physical device advertises it. The extension is passive — enabling it only causes the driver to populate `VkPhysicalDeviceMemoryBudgetPropertiesEXT` on `vkGetPhysicalDeviceMemoryProperties2` calls — so this is observable in `Device::enabled_extensions()` but has no runtime cost when unused. Opt-out is not currently exposed; file an issue if you need it.
- `Allocator::query_budget()` doc comment clarified: budget numbers are meaningful when `has_memory_budget_support()` returns `true`, which the auto-enable path makes the default on supported drivers.

## [0.6.0] — 2026-04-16

Major version: every Vulkan extension and feature bit is now reachable from safe code via generated builders. Layer 1 + Layer 2 + Layer 3 of the extension-handling architecture are all landed, plus **Phase 1 of Layer 4** — RAII wrappers for every previously-unwrapped Vulkan handle type.

### Added

- **`PNextChainable` trait and `PNextChain` builder** (Layer 2) — a generic pNext-chain mechanism replaces every hand-rolled pointer-patching site in the crate.
  - `PNextChainable` is implemented by the generator for every `#[repr(C)]` struct in `vk.xml` whose first two fields are `sType: VkStructureType` and `pNext` — **1225 impls** emitted from the current spec.
  - `PNextChain` owns heap-stable boxed nodes, relinks `pNext` pointers on push, and supports typed read-back (`get::<T>()` / `get_mut::<T>()`) for output-direction queries like `vkGetPhysicalDeviceMemoryProperties2` + `VK_EXT_memory_budget`.
  - Every ad-hoc pNext site in `vulkane` has been rewritten to use the chain (device creation, queue submit, semaphore create, memory allocate, memory-budget query).
- **Generated `DeviceFeatures`** (Layer 1) — `vulkan_gen` now emits one `with_<feature>()` builder method per unique feature bit across every struct that extends `VkPhysicalDeviceFeatures2`. **541 feature-bit methods** generated from the current spec. Name collisions between core-aggregate structs (`VkPhysicalDeviceVulkan12Features`) and promoted/extension structs (`VkPhysicalDeviceTimelineSemaphoreFeaturesKHR`) are resolved by routing the method to the highest-priority struct; the other path remains reachable via `chain_extension_feature()`.
- **Generated `DeviceExtensions` / `InstanceExtensions`** (Layer 3) — one `<vendor>_<ext>()` enable-method per non-disabled extension, with transitive `requires` resolved at generation time. **416 device + 44 instance** methods emitted from the current spec. Fresh extensions not yet in your copy of `vk.xml` are reachable through `enable_raw(name)`.
- **Generated RAII handle wrappers** (Layer 4 — Phase 1) — one safe, Drop-aware wrapper for every Vulkan handle type whose Create / Destroy pair fits the standard four-/three-parameter shape and isn't already covered by a hand-written wrapper. **25 new safe types** in `vulkane::safe::auto`, including `AccelerationStructureKHR`, `AccelerationStructureNV`, `MicromapEXT`, `VideoSessionKHR`, `VideoSessionParametersKHR`, `DeferredOperationKHR`, `DescriptorUpdateTemplate`, `PrivateDataSlot`, `ValidationCacheEXT`, `BufferView`, `SamplerYcbcrConversion`, `IndirectCommandsLayoutEXT/NV`, `IndirectExecutionSetEXT`, and more. Creating or destroying any of these previously required `unsafe { dispatch().vk… }` — now it's one safe call with automatic cleanup on drop.
- **`Allocation` now implements `Drop`** — a forgotten `allocator.free()` no longer leaks the slot in the TLSF pool. `AllocationInner` carries a `Weak<AllocatorInner>` back-reference, so the slot is reclaimed when the last `Arc<AllocationInner>` clone goes out of scope. `Allocator::free(allocation)` is kept for callers who prefer the imperative style — it now just `drop`s. `vulkane::safe::MemoryRequirements` is now re-exported from the crate root so call sites that build it directly don't need to reach into the buffer module.
- **Generated safe-method ext traits for every Vulkan command** (Layer 4 — Phase 2) — **600 safe methods** across 5 ext traits (`DeviceExt` 237, `CommandBufferRecordingExt` 266, `PhysicalDeviceExt` 78, `QueueExt` 15, `InstanceExt` 4). Every Vulkan command with a recognizable dispatch target now has a safe method — no `unsafe { dispatch().vkX.unwrap()(…) }` required anywhere in user code. Methods keep the `vk_` prefix (e.g. `vk_cmd_trace_rays_khr`), take raw Vulkan parameter types, and return `Result<VkResult>` for VkResult-returning commands (with error codes in `Err`, success codes like `VK_INCOMPLETE` / `VK_SUBOPTIMAL_KHR` in `Ok`). Users opt in per trait: `use vulkane::safe::CommandBufferRecordingExt;`. Ergonomic sugar (slice collapsing, typed output params, enumerate helpers) deferred to a future polish pass.
- `camel_to_snake` helper in `vulkan_gen::codegen` for consistent Vulkan identifier → Rust method-name conversion across generators.

### Breaking

- **`DeviceCreateInfo::enabled_extensions` is now `Option<&DeviceExtensions>`** (previously `&[&str]`). Migrate:

  ```rust
  // before
  let exts = [KHR_SWAPCHAIN_EXTENSION];
  DeviceCreateInfo { enabled_extensions: &exts, .. }
  // after
  let exts = DeviceExtensions::new().khr_swapchain();
  DeviceCreateInfo { enabled_extensions: Some(&exts), .. }
  ```

- **`InstanceCreateInfo::enabled_extensions` is now `Option<&InstanceExtensions>`** (previously `&[&str]`). Same migration pattern.
- **`DeviceFeatures` fields and hand-written builder methods are gone**, replaced by the 541 generated `with_<feature>()` methods. Callers who were constructing `DeviceFeatures { features11: …, features12: …, .. }` manually should use the builder instead. The generator picks names identical to pre-existing ones (`with_timeline_semaphore`, `with_buffer_device_address`, …) so most call sites are unaffected.
- **Hand-written extension-name constants removed** (`KHR_SURFACE_EXTENSION`, `KHR_SWAPCHAIN_EXTENSION`, `DEBUG_UTILS_EXTENSION`, `EXT_METAL_SURFACE_EXTENSION`, `KHR_WIN32/WAYLAND/XLIB/XCB_SURFACE_EXTENSION`). Use the generated `crate::raw::bindings::<NAME>_EXTENSION_NAME` constants or (preferred) the `<vendor>_<ext>()` builder methods.
- **`PNextChainable` requires `Clone + Default + 'static`** (previously `Default + 'static`). All `vk.xml`-generated structs derive `Clone`, so this is only a source-level break for code that hand-implemented the trait.

## [0.5.0] — 2026-04-16

### Added

- **`ShaderRegistry` for precompiled SPIR-V shaders** — new `vulkane::safe::shaders` module providing a small, shared abstraction for applications that ship compiled `.spv` artifacts (embedded via `include_bytes!` and/or loaded from disk).
  - `ShaderSource { name: &'static str, spv: &'static [u8] }` — one entry per compiled shader.
  - `ShaderRegistry::new().with_embedded(&[...]).with_env_override("MY_APP_OVERRIDE_DIR")` — builder-style setup.
  - `registry.load(name) -> Cow<'_, [u8]>` — bytes.
  - `registry.load_words(name) -> Vec<u32>` — SPIR-V word layout.
  - `registry.load_module(&device, name) -> ShaderModule` — full device-bound module in one call.
  - Runtime disk override: if the configured env var points at a directory and `<dir>/<name>.spv` exists, it is loaded instead of the embedded default; otherwise the registry falls through to the embedded table. Ideal for shader developers iterating without rebuilding the whole binary.

### Breaking

- **`Error::ShaderLoad` payload changed from `String` to `ShaderLoadError`.** The old variant preserved only a string description; the new one carries a structured enum (`NotFound` / `Io { name, source }` / `MalformedSpirv { name, byte_len }`) so consumers can match on the failure reason. Migration for manual constructions: convert `Error::ShaderLoad(format!("..."))` into the matching `ShaderLoadError` variant. Code that already used `From<ShaderLoadError> for Error` (via `?` on a `ShaderRegistry` call) needs no changes.

## [0.4.5] — 2026-04-15

### Added

- **Optional `slang` feature** — runtime Slang → SPIR-V compilation via the `shader-slang` crate (Khronos Slang compiler). Slang adds modules, generics, interfaces, and — most relevant for ML compute on Vulkan — built-in automatic differentiation: tag a function `[Differentiable]` and request forward and backward entry points from the same compiled module.
  - `vulkane::safe::slang::SlangSession::{new, with_search_paths, load_file}` — session-based API for compiling one module into many entry-point SPIR-V blobs.
  - `vulkane::safe::slang::SlangModule::compile_entry_point(name) -> Result<Vec<u32>, SlangError>`.
  - `vulkane::safe::slang::compile_slang_file(search_dir, module, entry)` — one-shot convenience.
  - Re-exports `CompileTarget`, `OptimizationLevel`, `Stage` from `shader-slang`.
  - New `Error::SlangCompile(String)` variant bridged from `SlangError`.
  - Requires `VULKAN_SDK` (SDK ships `slangc`) or `SLANG_DIR` at build/link time; `slang.dll` / `libslang.so` must be on the runtime library search path.
  - **Current limitation**: `shader-slang` 0.1.0 does not expose inline source compilation; Slang modules must live in `.slang` files resolved through session search paths. Will be lifted when a newer `shader-slang` ships.

## [0.4.4] — 2026-04-15

### Documentation

- Sync the crate-level `vulkane/README.md` (shown on crates.io and docs.rs) with the repo-root README: add the `shaderc` feature entry, the runtime-shader-compilation section, the naga-vs-shaderc selection table, and shaderc build requirements. The 0.4.3 release updated only the repo-root copy.

## [0.4.3] — 2026-04-15

### Added

- **Optional `shaderc` feature** — runtime GLSL/HLSL → SPIR-V compilation via the Khronos reference `glslang` compiler (wrapped by `shaderc-rs`). Complements the existing `naga` feature for cases that need full GLSL extension support, HLSL input, or glslang-only optimization passes.
  - `vulkane::safe::shaderc::compile_glsl(source, kind, file_name, entry_point) -> Result<Vec<u32>, ShadercError>` — common case.
  - `vulkane::safe::shaderc::compile_with_options(..., |opts| { ... })` — HLSL input, optimization level, macro defines, include callbacks, target Vulkan version.
  - Re-exports `ShaderKind`, `SourceLanguage`, `TargetEnv` from `shaderc`.
  - New `Error::ShadercCompile(String)` variant bridged from `ShadercError`.
  - Build requires either the LunarG Vulkan SDK (`VULKAN_SDK` env var), a system libshaderc, or a C++ build toolchain (CMake + Python + C++ compiler) for the source-build fallback. See README for details.

## [0.4.0] — 2026-04-10

### Added

- **45 Format constants** (up from 11) covering 8/16/32-bit, depth, and BC compressed formats. No more reaching into `vulkane::raw::bindings::VkFormat` for vertex attribute formats.
- **`Format::bytes_per_pixel()`** — returns the byte size per pixel for common uncompressed formats.
- **`BufferCopy::full(size)`** — one-liner for the common offset-0 copy case.
- **`#[derive(Vertex)]` proc macro** (new `vulkane_derive` crate, opt-in via `derive` feature) — auto-generates `VertexInputBinding` + `VertexInputAttribute` from `#[repr(C)]` structs. Supports `f32`, `[f32; 2..4]`, `u32`, `[u32; 2..4]`, `i32`, `[i32; 2..3]`, `[u8; 4]`, `u16`, `i16`. Provides both `::binding()` (vertex rate) and `::instance_binding()` (instance rate).
- New example: `derive_vertex` — instanced triangles using the derive macro.

## [0.3.0] — 2026-04-10

### Added

- **Pipeline builder extensions:**
  - `depth_bias(constant, slope, clamp)` — shadow acne prevention.
  - `depth_compare_op(CompareOp)` with `CompareOp` enum (NEVER / LESS / EQUAL / LESS_OR_EQUAL / GREATER / NOT_EQUAL / GREATER_OR_EQUAL / ALWAYS).
  - `InputRate` (VERTEX / INSTANCE) on `VertexInputBinding` — instanced rendering.
  - `color_attachment_count(n)` — multi-attachment / G-buffer pipelines.
  - `dynamic_viewport_scissor()` — resize-friendly pipelines with `set_viewport` / `set_scissor`.
- **Depth image views** — `ImageView::new_2d_depth` for depth-aspect views.
- **Image barrier aspect mask** — `ImageBarrier` gains `aspect_mask` field + `::color()` / `::depth()` convenience constructors.
- **`ClearValue` enum** + `begin_render_pass_ext` for mixed color + depth/stencil clear values.
- **Comparison sampler** — `SamplerCreateInfo::compare_op` for shadow map sampling.
- **Allocation helpers:**
  - `Buffer::new_bound(device, physical, info, flags)` — 5-step boilerplate → 1 call.
  - `Image::new_2d_bound(device, physical, info, flags)` — same for images + auto color view.
  - `Queue::upload_buffer<T>(device, physical, qf, data, usage)` — staging upload in one call.
  - `Queue::upload_image_rgba(device, physical, qf, w, h, pixels)` — image upload with layout transitions.
- New examples: `depth_prepass`, `instanced_mesh`, `shadow_map`, `deferred_shading`.

### Breaking

- `ImageBarrier` now requires `aspect_mask: u32` field. Use `ImageBarrier::color(...)` or `ImageBarrier::depth(...)` constructors.

## [0.2.0] — 2026-04-09

### Added

- **Typed pipeline stage and access mask constants** — `PipelineStage`, `AccessFlags` (32-bit), `PipelineStage2`, `AccessFlags2` (64-bit for Sync2). All barrier, timestamp, and sync APIs now accept these types instead of raw `u32` / `u64`.
- **Convenience constructors:**
  - `QueueCreateInfo::single(family_index)` — one queue, priority 1.0.
  - `WaitSemaphore::binary(sem, stage)` / `::timeline(sem, value, stage)`.
  - `SignalSemaphore::binary(sem)` / `::timeline(sem, value)`.
  - `RenderPass::simple_color(device, format, load, store, final_layout)`.
  - `Queue::one_shot(device, qf, |rec| { ... })` — fire-and-forget command recording.
- **Raw escape hatch** — `Device::dispatch()` and `Instance::dispatch()` expose the full dispatch tables for calling any Vulkan function alongside safe wrapper types.
- New examples: `buffer_upload`, `raw_interop`, `allocator_compute`.

### Breaking

- All barrier/sync API signatures changed from raw `u32`/`u64` to typed `PipelineStage`/`AccessFlags`. Migration: `0x800` → `PipelineStage::COMPUTE_SHADER`.
- `WaitSemaphore::dst_stage_mask` changed from `u32` to `PipelineStage`.

## [0.1.0] — 2026-04-08

### Added

- Initial release: complete Vulkan bindings generated from vk.xml + safe RAII wrapper covering compute and graphics end-to-end.
- **Raw bindings** (`vulkane::raw`) — all types, enums, structs, function pointers, and three-tier dispatch tables generated from the spec.
- **Safe wrapper** (`vulkane::safe`) — RAII handles for Instance, Device, Buffer, Image, ImageView, Sampler, DeviceMemory, ShaderModule, DescriptorSetLayout/Pool/Set, PipelineLayout, ComputePipeline, GraphicsPipeline (with builder), RenderPass, Framebuffer, Surface (Win32/Wayland/Xlib/Xcb/Metal), Swapchain, CommandPool/Buffer, Fence, Semaphore (binary + timeline), QueryPool.
- **VMA-style sub-allocator** — TLSF + linear pools, custom user pools, dedicated allocations, persistent mapping, defragmentation, memory budget queries.
- **Device groups** — unified single/multi-GPU device representation with per-allocation and per-submission device masks.
- **DeviceFeatures builder** — Vulkan 1.0/1.1/1.2/1.3 feature chain construction.
- **Optional `naga` feature** — `compile_glsl` + `compile_wgsl` → SPIR-V at runtime.
- **`fetch-spec` feature** — auto-download vk.xml from Khronos GitHub.
- 7 bundled examples: device_info, fill_buffer, compute_square, compute_image_invert, compile_shader, headless_triangle, textured_quad, windowed_triangle.
- Tree-based XML parser (roxmltree), vk.xml api-attribute filtering, VKSC profile exclusion.
- CI on Linux/Windows/macOS with Mesa Lavapipe for headless GPU tests.
