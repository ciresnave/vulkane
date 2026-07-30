# To the Fuel team — Kernel-Seam Interop Contract (Profile v1), Vulkane BDA-subset confirm

**Status: SENT (2026-06-28) — resolved.** Vulkane's reply to `docs/specs/kernel-seam-interop.md`
(Profile v1) and the `vulkane-seam-v1-confirm` circulation cover note.

**Outcome:** the contract was **ratified 2026-06-20** with all three wording fixes below folded into
§7.2 — which now states the Vulkane contract as **behavior + named surface** rather than a
`≥ 0.8.2` crate-version floor — plus the alignment-ownership clarification. Two follow-up nits were
corrected in the same pass (a dangling `§4.J` reference → §4.1, and §8 step 4's "confirms ≥ 0.8.2"
wording). No Vulkane library code was required: the entire named surface shipped in 0.8.2. Vulkane
subsequently added [`tests/profile_v1_conformance.rs`](../tests/profile_v1_conformance.rs), a
compile-time lock-in that makes any rename, removal, or signature change of that surface fail
Vulkane's CI rather than `fuel-vulkan-backend`'s build (shipped in 0.8.3). Retained as the record of
what Vulkane confirmed and why.

---

## Verdict

**Confirmed — both asks, yes.** The BDA subset maps to **FDX v1 unchanged** on our end, and the
light FDX-version handshake path (contract §3.5) is fine — it asks nothing of the Vulkane FFI, which
is exactly why we can confirm it cleanly. The work the contract pins on is **shipped**: `vulkane`
**0.8.2** (2026-06-19, on crates.io) is the device-address-capable allocator the `≥ 0.8.2` floor
refers to, so Vulkane is conformant to Profile v1 as-is.

We're confirming **conditioned on three wording fixes** below. None are design objections — the
design is right and the Vulkane subset is correctly minimal. They keep the contract accurate and
future-proof so a later reader (or a later Vulkane release) doesn't trip on a silent assumption.

## What we confirm (all from the 2026-06-19 freeze, now shipped)

- `data = VkDeviceAddress` on `kDLVulkan`; `byte_offset` stays a separate wire field; the backend
  folds `data + byte_offset` at dispatch in `fuel-vulkan-backend`. `Buffer::device_address()`
  ([`vulkane/src/safe/buffer.rs:266`](../src/safe/buffer.rs)) returns exactly that base, 1:1.
- Block-level device-address support shipped: `Allocator::new_with_options(.., AllocatorOptions {
  buffer_device_address: true })` puts `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` on every block
  ([`vulkane/src/safe/allocator/mod.rs:419`](../src/safe/allocator/mod.rs)), so addresses read from
  pooled / `new_bound` buffers are valid on strict drivers.
- Sidecar stays Fuel-side (no Vulkane ABI slot); signed strides never reach a binding; plural,
  descriptor-set-free buffer table; the 6-dim shape/stride cap is ignored on our side.
- **No FKC, no JIT, no kernels.** Vulkane ships none; Fuel's Vulkan kernels are internal Slang. Our
  Slang/shaderc surface is kernel *tooling*, not dispatchable entry points we ship — so FDX-only is
  correct. (If Vulkane ever exposes its own compute entry points Fuel could dispatch to, *those*
  would carry FKC contracts at that point — out of scope for Profile v1, agreed.)

## Three requested wording fixes

### 1. State the BDA "subset" as its three caller preconditions, not as an automatic guarantee

The contract (§7.2) and the Vulkane cover note say "`SHADER_DEVICE_ADDRESS` on every buffer-table
entry" as if Vulkane applies it. It doesn't — Vulkane never auto-applies it. For
`Buffer::device_address()` to return a valid address, **`fuel-vulkan-backend` must satisfy all three
of these together**, or it silently gets a wrong or erroring address:

1. Construct the allocator via `new_with_options(.., buffer_device_address: true)` — else pooled
   buffers' addresses are invalid on strict drivers (the exact bug 0.8.2 fixed).
2. Create **each buffer-table buffer** with `BufferUsage::SHADER_DEVICE_ADDRESS`
   ([`buffer.rs:62`](../src/safe/buffer.rs)) — this is per-buffer usage, not an allocator-wide
   property.
3. Enable the `bufferDeviceAddress` device feature at device creation via
   `DeviceFeatures::with_buffer_device_address()` (Vulkan 1.2 core) — else `vkGetBufferDeviceAddress`
   isn't loaded and `device_address()` returns `Error::MissingFunction`, not a bad value.

Please list these as the BDA-subset preconditions in §7.2. They're a producer-side obligation, and
the "never silent coercion" discipline the contract champions should name them rather than leave
them implicit.

### 2. §3.5 misattributes the handshake to Vulkane

§3.5 says "Fuel reads Vulkane's advertised max FDX version via `BackendCapabilities` … no new FFI on
the Vulkane side." The Vulkane FFI crate exposes **no** `BackendCapabilities` / `BackendProbe` — those
are Fuel-side FDX abstractions. **Nothing on the wire originates from Vulkane.** In practice
`fuel-vulkan-backend` (Fuel-side glue) advertises the FDX version, derived from the linked `vulkane`
crate version.

This is actually *better* than the spec implies — Vulkane does literally nothing, so there's nothing
on our side to version or break. Please reword to "`fuel-vulkan-backend` advertises, derived from the
linked `vulkane` version" rather than "Vulkane advertises," so a future reader doesn't hunt for a
Vulkane FFI version entry point that doesn't and won't exist.

### 3. Pin the Vulkane contract to behavior + named surface, not the crate version number

`vulkane ≥ 0.8.2` is a fine *informative* floor, but it shouldn't be the *normative* contract. Our
version bumps for reasons unrelated to this seam — 0.8.0 → 0.8.1 → 0.8.2 inside two months
(Send + Sync markers, then the allocator). If 0.9 / 1.0 ships an unrelated breaking change,
"≥ 0.8.2" silently keeps asserting Profile-v1 conformance even though the named API the backend calls
may have moved.

Please state the Vulkane contract as the **behavior + named surface** — `data = VkDeviceAddress` on
`kDLVulkan`, `byte_offset` folded at dispatch, block-level `DEVICE_ADDRESS` + per-buffer
`SHADER_DEVICE_ADDRESS`, via `AllocatorOptions::buffer_device_address` / `Buffer::device_address` —
with "≥ 0.8.2 = first version exposing it." Then a Vulkane major bump triggers a re-check of that
named surface instead of a silent pass.

## One minor clarification (not a blocker)

"The 256-byte floor dominates" (§7.2): Vulkane honors the buffer's `VkMemoryRequirements.alignment`
(storage buffers: `minStorageBufferOffsetAlignment`, often 16–256) — **not** a guaranteed
256-byte-aligned base address. Since `byte_offset` is folded at dispatch and Fuel owns it, only the
*final* `data + byte_offset` must meet the kernel's load alignment, which Fuel governs. So this is a
non-issue **as long as** the contract reads "Fuel ensures final-address alignment," not "Vulkane
guarantees 256-aligned base addresses." One line pinning which side owns it would close the gap.

(§4.2 footnote 2 — "Vulkane carries the buffer-table roles structurally but does not interpret them"
— is accurate as written; we're role-agnostic, the DATA/SCALE/ZERO_POINT/… role→index mapping is
entirely `fuel-vulkan-backend`'s. No change there.)

## Net

Vulkane confirms Profile-v1 conformance at `≥ 0.8.2`. Make the §1/§3.5/§7.2 wording fixes above and
we're ready to ratify alongside Baracuda. Thanks — the BDA design is doing exactly what we hoped at
the seam.
