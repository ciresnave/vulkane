# To the Fuel team — device-load telemetry (Step E, Phase B2)

**Status: SENT (2026-06-28).** Vulkane's reply to the *device-load telemetry* ask
(`fuel/docs/session-prompts/step-e-async-execution.md`, Phase B). The `device_identity()` surface it
describes shipped in **vulkane 0.8.3** (released 2026-06-28, on crates.io). Retained as the record of
where the telemetry boundary was drawn and why.

---

## Verdict

**Candidate 1 (`pending_submissions`): declined — nothing to expose.**
**Candidate 2 (cross-process device load): not a Vulkan capability, so not a Vulkane one — but we've
shipped the one piece that legitimately *is* ours: device identity, the join key you need to go ask
an out-of-band source.**

Vulkan provides **no** cross-process — or even in-process — GPU load / utilization / queue-depth
query. The only device-state telemetry in the API is VRAM (`VK_EXT_memory_budget`, which you already
consume via `Allocator::vram_budget` / `vram_used`). There is no compute-load analog anywhere in core
or KHR. So the honest answer to "is any cross-process busy signal reachable *from Vulkane*?" is **no**,
and that's a boundary of Vulkan, not a gap in Vulkane.

## Candidate 1 — `pending_submissions()`: declined

Your caveat was "unless vulkane internally queues submissions fuel can't see." It doesn't.
`Queue::submit*` is a direct `vkQueueSubmit` passthrough
([`device.rs:560`](../src/safe/device.rs) → [`device.rs:621`](../src/safe/device.rs)) — one submit,
immediately, no buffering. `Queue::one_shot` ([`device.rs:798`](../src/safe/device.rs)), the
"batch + fence per flush" pattern you were picturing, records → submits → **waits** → frees, all
synchronously inside the call; nothing survives it. Fences are caller-owned (passed *into* `submit`,
never retained), so Vulkane keeps no central submission or fence registry and holds **zero**
cross-call state you can't see.

A `pending_submissions()` would therefore return a constant 0 — a lie dressed as a signal. Your
async in-flight counter (Step E Phase A) is authoritative and complete. Keep it as the sole owner of
this; we won't add the method.

## Candidate 2 — cross-process device load: identity is the seam, load lives elsewhere

Every real busy-signal source keys off **vendor or OS**, never off Vulkan: NVML (NVIDIA), Windows PDH
"GPU Engine" counters / D3DKMT, Linux `amdgpu` sysfs `gpu_busy_percent`, i915/Xe perf, IOKit/Metal on
macOS. The number is the *same* whether the work reached the GPU through Vulkan, CUDA, D3D, or Metal.
Binding a telemetry backend into a `vk.xml`-generated Vulkan wrapper would drag vendor/OS deps into a
crate whose whole identity is "Vulkan, from spec," and would deny the signal to every non-Vulkan
consumer. So the load lookup is **not a Vulkane concern**, and we're deliberately not putting it here.

What *is* ours — the only Vulkan-shaped part of this problem — is telling you **which physical GPU you
are holding**, in terms an out-of-band source can match. That shipped:

### `PhysicalDevice::device_identity() -> Option<DeviceIdentity>` (Vulkane 0.8.3)

*(Unreleased when this reply was written; released as 0.8.3 on 2026-06-28.)*

One `vkGetPhysicalDeviceProperties2` call returning:

| Field | Source | Match target | Populated when |
|---|---|---|---|
| `device_uuid: [u8; 16]` | `VkPhysicalDeviceIDProperties` (1.1 core) | NVML (`nvmlDeviceGetUUID`), CUDA, OpenGL | always (props2 present) |
| `driver_uuid: [u8; 16]` | same | ICD disambiguation | always |
| `device_luid: Option<[u8; 8]>` + `device_node_mask: u32` | same (`deviceLUIDValid`) | DXGI adapter / D3DKMT node | `Some` on Windows only |
| `pci: Option<PciBusInfo>` (`domain:bus:device.function`) | `VK_EXT_pci_bus_info` | Linux `/sys/bus/pci/devices/…` → `gpu_busy_percent` | `Some` when the device advertises the ext |

Same contract discipline as the VRAM query: `Send + Sync`, allocation-free, `Option`-returning, never
panics, and **honest `None` at every level** — `None` if props2 is unavailable, `device_luid: None`
off Windows, `pci: None` on software rasterizers. (Our CI's software rasterizer returns all-zero UUID,
no LUID, no PCI — the honest-`None` paths verified.) New public types `safe::DeviceIdentity` /
`safe::PciBusInfo`. Non-breaking, additive.

### Recommended division of labor

1. **Vulkane** gives you the identity (done). That's the full extent of our role here.
2. **An API-agnostic GPU-load crate** owns the vendor/OS backends (NVML / PDH / D3DKMT / sysfs / …)
   behind one `Option`-returning, identity-keyed, alloc-free `Send + Sync` query — the same contract
   shape, just not in Vulkane. We'd suggest it lives Fuel-side for now (you're the only consumer and
   `BackendRuntime` already abstracts backends, so the same crate serves your CUDA/other backends),
   designed neutrally so it can spin out as a standalone crate if a second consumer appears. Reuse
   beats building — worth a crates.io scan first; the mature pieces are vendor-specific
   (`nvml-wrapper`, `amdgpu-sysfs`) and we're not aware of a maintained unified facade.
3. **`DeviceLoadSelector`** joins them: `vulkane.device_identity()` → telemetry crate → `Option<load>`.

This keeps your stated constraints intact — no new *required* dep on Vulkane's side (identity is core
+ one optional ext, zero new deps), read-only, never stalls the queue, honest `None`. And it matches
your own framing: if no portable cross-process signal exists, ship Step E on the internal counter and
treat cross-process visibility as a backlog item layered on top — now with a real identity hook to
build it against.

## Net

`pending_submissions`: no. Cross-process load via Vulkan: no — the API doesn't have it. Device
identity for out-of-band correlation: shipped as `device_identity()`. The load layer is a separate,
API-agnostic concern we recommend you own; Vulkane stays a Vulkan wrapper and provides exactly the
Vulkan-shaped half of the seam.
