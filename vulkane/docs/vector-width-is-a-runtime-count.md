# Vector width is a runtime COUNT, per device — not a static byte cap

Recorded because the question was asked, measured, and answered, and the answer
retired three proposed designs. Without this note the next person reaching for a
per-target vector-width constant re-derives it from scratch — and the plausible
wrong answer is cheaper to reach than the right one.

**Asked (2026-09-03, by the Unpopped lane):** what is the maximum SPIR-V/Vulkan
vector width, so it can be expressed as a static per-target constant in bytes?

**Answered: no such constant is expressible.** Three independent reasons, any one
of which is sufficient.

## 1. The unit was wrong — it is a count of COMPONENTS, not bytes

`VK_EXT_shader_long_vector` reports
`VkPhysicalDeviceShaderLongVectorPropertiesEXT.maxVectorComponents`. Components,
not bytes. A cap in bytes is not a rescaling of it: the byte size depends on the
component type, so the same limit is a different byte figure for `f16`, `f32`
and `f64`. **A byte-valued constant cannot represent this quantity at all**, so
the design question was not "which number" but "which quantity".

## 2. It is queried at runtime, not known at compile time

It arrives through a `pNext` chain on `vkGetPhysicalDeviceProperties2`. There is
no header constant, no target feature, and nothing a build script can read.

## 3. It is PER DEVICE, and one machine can disagree with itself

Measured on a single machine with two GPUs (2026-09-03):

    AMD Radeon 610M (integrated)   VK_EXT_shader_long_vector  ABSENT
    NVIDIA RTX 4070 Laptop         present, maxVectorComponents = 1024

**A per-target constant would have to hold for both, and there is no value that
does.** "Target" is the wrong axis: the quantity varies per *device*, selected at
run time, on hardware a build never sees.

## The hazard attached to measuring it

An **ungated** `pNext` read of that struct on the 610M returns **zero**, because
a driver that does not implement the extension leaves the chained struct exactly
as the caller allocated it. `PNextChain::get` then hands it back
indistinguishably from a real answer.

The registry classifies it that way itself, so this is not a reading of mine:
`vk.xml` marks the struct `returnedonly="true" requiredlimittype="true"` and the
member `limittype="max"`. **It is a limit by the spec's own taxonomy**, and a zeroed
limit is a false reading rather than a conservative one.

**Zero components is a plausible number and a false one.** It is not "no long
vectors" — it is a struct nobody wrote. Check the extension is present, or gate
on the effective API version, before believing any limit read this way. The
general form of this trap, and why it is worse for limits than for capability
booleans, is documented on
[`PhysicalDevice::ray_tracing_pipeline_properties`][rt] and was the subject of
the `device_identity` fix in 0.15.0, where the same shape produced a device UUID
that compared equal across physically different GPUs.

[rt]: ../src/safe/physical.rs

## What vulkane exposes

**Nothing, currently.** There is no safe accessor for
`VkPhysicalDeviceShaderLongVectorPropertiesEXT`; the measurement above was taken
with an ad-hoc probe against the raw bindings. That is a deliberate non-decision
rather than an omission with a plan behind it: no caller in this workspace needs
it yet, and adding an accessor whose only user is a question that has already
been answered would be surface with no consumer.

**If one is added**, it belongs with the other property queries in
`src/safe/physical.rs` and must gate the way its neighbours do — return `None`
when the extension is absent rather than a struct full of zeros, because
`maxVectorComponents` is a *limit* and a zeroed limit is a false reading rather
than a conservative one.
