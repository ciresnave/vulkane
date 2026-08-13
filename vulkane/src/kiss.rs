//! Derive KISS-Classify §6.8 `vulkan:` `target_capability` tokens from a live
//! physical device.
//!
//! This is the **deriver** half of the `vulkan:` namespace. The other half —
//! the vocabulary itself (canonical spelling, parsing, comparison) — lives in
//! the dependency-free [`kiss_vulkan_vocab`] crate, because
//! KISS-CLASSIFY-6.9-0003 forbids producing or parsing a token from loading a
//! compute driver. Deriving one from a real `VkPhysicalDevice` obviously
//! *does* need Vulkan, so it lives here instead, where that dependency already
//! exists. A conformance implementation needs only the vocabulary crate; this
//! module is for a caller that actually has a GPU in hand.
//!
//! # A device does not have a token — it *admits* a set of them
//!
//! A `target_capability` names the specialization a kernel was **built for**,
//! not the capability envelope of the device that runs it (the same way
//! `cuda:sm89` names what a kernel was compiled for, and an Ada part runs
//! `sm_80` / `sm_86` / `sm_89` binaries alike). On a device reporting a
//! pinnable subgroup range of 32..=64, a wave32-pinned kernel and a
//! wave64-pinned kernel are different binaries and therefore different cells;
//! a token naming the envelope would collide them.
//!
//! So the API here is deliberately **not** `device -> token`. It is:
//!
//! - [`DeviceCapabilities`] — what the device offers (the envelope),
//! - [`DeviceCapabilities::admissible_subgroups`] — the axis a caller chooses
//!   along,
//! - [`DeviceCapabilities::target_for`] — one concrete token per choice,
//! - [`DeviceCapabilities::admits`] — whether a device can run a given cell.
//!
//! ```no_run
//! use vulkane::safe::*;
//! use vulkane::kiss::DeviceCapabilities;
//!
//! // Note the fully-qualified Result: the glob above brings Vulkane's own
//! // one-parameter `Result<T>` alias into scope, which shadows std's.
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! // 1.2 or higher: the arithmetic and driver queries are gated on the
//! // INSTANCE version, not the device's.
//! let instance = Instance::new(InstanceCreateInfo {
//!     api_version: ApiVersion::V1_3,
//!     ..Default::default()
//! })?;
//! let physical = instance.enumerate_physical_devices()?.remove(0);
//!
//! if let Some(caps) = DeviceCapabilities::of(&physical) {
//!     for sg in caps.admissible_subgroups() {
//!         println!("{}", caps.target_for(sg).to_token());
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use crate::raw::bindings::VkComponentTypeKHR;
use crate::safe::{PhysicalDevice, SubgroupFeatureFlags};
use kiss_vulkan_vocab::{
    Arith, ComponentType, CoopMatrix, CoopShape, OpClasses, Subgroup, VulkanTarget,
};

/// What a physical device offers, in `vulkan:` vocabulary terms.
///
/// This is the **envelope**, not a specialization. Use
/// [`admissible_subgroups`](Self::admissible_subgroups) to enumerate the
/// choices it permits and [`target_for`](Self::target_for) to spell one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    /// The device's default subgroup width.
    pub default_subgroup: u32,
    /// The pinnable width range, when the device exposes subgroup-size
    /// control. `None` means the width is fixed at
    /// [`default_subgroup`](Self::default_subgroup).
    pub subgroup_range: Option<(u32, u32)>,
    /// Subgroup operation classes the device implements.
    pub ops: OpClasses,
    /// Arithmetic capabilities the device implements.
    pub arith: Arith,
    /// Cooperative-matrix shapes the device supports, canonically ordered.
    pub coop: Vec<CoopShape>,
}

impl DeviceCapabilities {
    /// Read a device's capability envelope.
    ///
    /// Returns `None` when the device cannot answer honestly — principally
    /// when [`PhysicalDevice::subgroup_properties`] declines, which happens on
    /// an [`Instance`](crate::safe::Instance) created below Vulkan 1.1 *however
    /// new the device is*, because an implementation must behave as the
    /// version the instance requested. Subgroup width is not optional here:
    /// a `vulkan:` token without it would name nothing useful, so a device
    /// that will not report it gets a decline rather than a guess.
    ///
    /// Every other axis degrades honestly — a device with no
    /// cooperative-matrix support yields an empty shape list, not a failure.
    pub fn of(physical: &PhysicalDevice) -> Option<Self> {
        let sg = physical.subgroup_properties()?;

        let mut ops = OpClasses::NONE;
        let supported = sg.supported_operations;
        for (flag, class) in [
            (SubgroupFeatureFlags::BASIC, OpClasses::BASIC),
            (SubgroupFeatureFlags::VOTE, OpClasses::VOTE),
            (SubgroupFeatureFlags::ARITHMETIC, OpClasses::ARITHMETIC),
            (SubgroupFeatureFlags::BALLOT, OpClasses::BALLOT),
            (SubgroupFeatureFlags::SHUFFLE, OpClasses::SHUFFLE),
            (
                SubgroupFeatureFlags::SHUFFLE_RELATIVE,
                OpClasses::SHUFFLE_RELATIVE,
            ),
            (SubgroupFeatureFlags::CLUSTERED, OpClasses::CLUSTERED),
            (SubgroupFeatureFlags::QUAD, OpClasses::QUAD),
            (SubgroupFeatureFlags::ROTATE, OpClasses::ROTATE),
            (
                SubgroupFeatureFlags::ROTATE_CLUSTERED,
                OpClasses::ROTATE_CLUSTERED,
            ),
            (SubgroupFeatureFlags::PARTITIONED_NV, OpClasses::PARTITIONED),
        ] {
            if supported.contains(flag) {
                ops |= class;
            }
        }

        let mut arith = Arith::NONE;
        if let Some(f) = physical.shader_arithmetic_features() {
            if f.shader_float16 {
                arith |= Arith::FLOAT16;
            }
            if f.shader_int8 {
                arith |= Arith::INT8;
            }
            if f.storage_buffer_16bit {
                arith |= Arith::STORAGE16;
            }
            if f.storage_buffer_8bit {
                arith |= Arith::STORAGE8;
            }
        }
        if physical
            .shader_integer_dot_product_properties()
            .is_some_and(|d| d.has_any_int8_acceleration())
        {
            arith |= Arith::DOT8;
        }

        let mut coop: Vec<CoopShape> = physical
            .cooperative_matrix_properties()
            .iter()
            .map(|p| CoopShape {
                m: p.m_size(),
                n: p.n_size(),
                k: p.k_size(),
                // Deliberately the `_raw` accessors. `component` maps an
                // unrecognized value to `ComponentType::Other(n)`, which needs
                // `n`; going through the checked accessor would collapse every
                // unknown type to `None` and lose the number that distinguishes
                // them. The token must stay honest about what the device said.
                a: component(p.a_type_raw() as u32),
                b: component(p.b_type_raw() as u32),
                c: component(p.c_type_raw() as u32),
                result: component(p.result_type_raw() as u32),
                saturating: p.saturating_accumulation(),
            })
            .collect();
        // Driver report order is not guaranteed stable across drivers or even
        // across calls, and the token must be byte-identical either way.
        coop.sort();
        coop.dedup();

        Some(Self {
            default_subgroup: sg.subgroup_size,
            subgroup_range: sg
                .size_control
                .map(|s| (s.min_subgroup_size, s.max_subgroup_size)),
            ops,
            arith,
            coop,
        })
    }

    /// Every subgroup specialization this device admits, canonically ordered.
    ///
    /// Always includes [`Subgroup::Dynamic`], since a width-agnostic kernel
    /// runs anywhere. Where the device exposes subgroup-size control, every
    /// power of two in the pinnable range is admissible; otherwise the single
    /// fixed width is.
    ///
    /// This is the choice axis: a caller picks one of these, and *that* is
    /// what the token names.
    pub fn admissible_subgroups(&self) -> Vec<Subgroup> {
        let mut out = vec![Subgroup::Dynamic];
        match self.subgroup_range {
            Some((min, max)) => {
                let mut w = min.max(1).next_power_of_two();
                while w <= max {
                    out.push(Subgroup::Fixed(w));
                    match w.checked_mul(2) {
                        Some(next) => w = next,
                        None => break,
                    }
                }
            }
            None => out.push(Subgroup::Fixed(self.default_subgroup)),
        }
        out
    }

    /// Spell the maximal target this device supports at the chosen subgroup
    /// specialization — every op class, arithmetic capability, and
    /// cooperative-matrix shape it offers.
    ///
    /// This is the *device's* full capability at that width. A kernel that
    /// uses less than all of it should spell its own narrower
    /// [`VulkanTarget`] directly, since the token names what the **kernel**
    /// requires; over-claiming would fragment cells that could otherwise be
    /// shared.
    pub fn target_for(&self, subgroup: Subgroup) -> VulkanTarget {
        VulkanTarget {
            subgroup,
            ops: self.ops,
            arith: self.arith,
            coop: CoopMatrix::from_shapes(self.coop.clone()),
        }
    }

    /// Whether this device can run a kernel built for `target`.
    ///
    /// Note this is a *capability* question, deliberately separate from token
    /// matching. KISS-CLASSIFY-6.8-0002 forbids a consumer from applying
    /// subset or implication logic when matching two tokens — so a consumer
    /// may **not** use this to decide that a `sg32` kernel matches its
    /// `sg64`-spelled cell. Use it to choose which cell to *build or request*;
    /// then match that cell's token byte-exactly.
    pub fn admits(&self, target: &VulkanTarget) -> bool {
        let width_ok = match target.subgroup {
            Subgroup::Dynamic => true,
            Subgroup::Fixed(w) => self.admissible_subgroups().contains(&Subgroup::Fixed(w)),
        };
        let coop_ok = match &target.coop {
            CoopMatrix::None => true,
            CoopMatrix::Shapes(s) => s.iter().all(|x| self.coop.contains(x)),
            // A digest cannot be checked against a shape list without
            // recomputing it, and answering "yes" on an unverifiable claim
            // would be worse than declining.
            CoopMatrix::Digest(_) => {
                matches!(
                    CoopMatrix::from_shapes(self.coop.clone()),
                    CoopMatrix::Digest(d) if CoopMatrix::Digest(d) == target.coop
                )
            }
        };
        width_ok && self.ops.contains(target.ops) && self.arith.contains(target.arith) && coop_ok
    }
}

/// Map a raw `VkComponentTypeKHR` to the vocabulary's component type.
///
/// `VkComponentTypeKHR` carries 16 values in the pinned `vk.xml` (Vulkan 1.4,
/// header 348): eleven in the base KHR set, five added by extension. All
/// sixteen are accounted for below, because an unmapped one is *invisible* —
/// it becomes `Other(n)`, which is a perfectly valid token, so nothing errors
/// and the device simply never matches a target naming that dtype.
///
/// - **0..=10 and `BFLOAT16_KHR`** — mapped.
/// - **`SINT8_PACKED_NV` / `UINT8_PACKED_NV`** — deliberately left `Other`.
///   These carry `s8`/`u8` data in a *packed* cooperative-matrix layout, which
///   is a different shader-side contract from the unpacked types. Folding them
///   onto `S8`/`U8` would collapse two distinct Vulkan values onto one token
///   and let a packed-only device satisfy a target asking for plain `s8`. An
///   honest `Other` beats a token claiming something the device doesn't offer.
/// - **`FLOAT8_E4M3_EXT` / `FLOAT8_E5M2_EXT`** — mapped as of vocabulary
///   version 3. Both blockers that previously stood here are now discharged,
///   and the second one is worth recording because it was resolved by argument
///   rather than by waiting.
///
///   The first was ordering: [`ComponentType`] had no FP8 variant, and adding
///   one had to ride the coordinated `sk4` schema event rather than precede it.
///   `sk4` Phase-0 has merged, and the enum is now `#[non_exhaustive]`.
///
///   The second was that these names denote a **layout**, the `fn`/`fnuz`
///   suffix is mandatory, and `vk.xml` says nothing about which is meant — so
///   the mapping looked like a guess between `f8e4m3fn` and `f8e4m3fnuz`, where
///   guessing wrong yields a silently wrong token rather than an error. That
///   was the right worry and it has an answer that does not depend on reading
///   prose: **KISS reserves the two `fnuz` spellings with no computation
///   semantics at all** — a `structure_key` using one must be met with a typed
///   decline (KISS-CLASSIFY-6.1-0001). Mapping a type a device *actually
///   computes with* onto a spelling defined to have no computation semantics
///   is incoherent, which leaves `f8e4m3fn` / `f8e5m2` as the only coherent
///   targets. Vulkan also exposes no `fnuz` enumerant, so no value can collide.
///   The layouts themselves are pinned normatively by KISS-OPS-6.16-0004 and
///   -0005 (OCP OFP8), which did not exist when this comment first said the
///   registry was silent.
///
///   `FLOAT_E4M3_NV` / `FLOAT_E5M2_NV` are *aliases* of the EXT enumerants —
///   the generator emits them as `pub const`s pointing at the same variant, so
///   there is one value per name and no second arm to write.
///
/// Values outside all of these become [`ComponentType::Other`] rather than an
/// error: new Vulkan component types appear faster than a vocabulary revision
/// can track them, and an honest round-trippable token beats a decline.
fn component(raw: u32) -> ComponentType {
    // Taken from the generated binding rather than written as a literal. The
    // base values are 0..=10 and check by eye; an extension value like
    // 1000141000 is derived from an extension number and an offset and does
    // not — so the one that could be silently wrong is the one the compiler
    // should own.
    const BFLOAT16: u32 = VkComponentTypeKHR::COMPONENT_TYPE_BFLOAT16_KHR as u32;
    const F8E4M3: u32 = VkComponentTypeKHR::COMPONENT_TYPE_FLOAT8_E4M3_EXT as u32;
    const F8E5M2: u32 = VkComponentTypeKHR::COMPONENT_TYPE_FLOAT8_E5M2_EXT as u32;

    match raw {
        0 => ComponentType::F16,
        1 => ComponentType::F32,
        2 => ComponentType::F64,
        3 => ComponentType::S8,
        4 => ComponentType::S16,
        5 => ComponentType::S32,
        6 => ComponentType::S64,
        7 => ComponentType::U8,
        8 => ComponentType::U16,
        9 => ComponentType::U32,
        10 => ComponentType::U64,
        BFLOAT16 => ComponentType::BF16,
        F8E4M3 => ComponentType::F8E4M3FN,
        F8E5M2 => ComponentType::F8E5M2,
        n => ComponentType::Other(n),
    }
}

#[cfg(test)]
mod component_tests {
    use super::*;

    /// The defect this test exists for: `ComponentType::BF16` and its `bf16`
    /// token have always existed in the vocabulary and round-trip correctly,
    /// so the vocabulary crate's tests pass — but they construct the variant
    /// directly. Nothing exercised the *derivation* from a raw device value,
    /// and `component()` had no `BFLOAT16_KHR` arm, so a driver reporting
    /// bfloat16 cooperative matrices yielded `Other(1000141000)`. Reachable by
    /// token, underivable from hardware.
    #[test]
    fn bfloat16_derives_from_the_raw_device_value() {
        assert_eq!(
            component(VkComponentTypeKHR::COMPONENT_TYPE_BFLOAT16_KHR as u32),
            ComponentType::BF16,
            "a device reporting VK_COMPONENT_TYPE_BFLOAT16_KHR must derive BF16, \
             not Other — otherwise bf16 is spellable but not derivable"
        );
    }

    /// Pins the whole base set, so a renumbering or a transposed arm is caught
    /// as a mismatch rather than as a token nobody notices is wrong.
    #[test]
    fn every_base_component_type_maps_to_its_documented_variant() {
        let expected = [
            (0, ComponentType::F16),
            (1, ComponentType::F32),
            (2, ComponentType::F64),
            (3, ComponentType::S8),
            (4, ComponentType::S16),
            (5, ComponentType::S32),
            (6, ComponentType::S64),
            (7, ComponentType::U8),
            (8, ComponentType::U16),
            (9, ComponentType::U32),
            (10, ComponentType::U64),
        ];
        for (raw, want) in expected {
            assert_eq!(component(raw), want, "VkComponentTypeKHR value {raw}");
        }
    }

    /// The packed NV types remain *deliberately* unmapped, for reasons recorded
    /// on [`component`]. This asserts the deliberate choice so that mapping them
    /// later is a decision someone makes on purpose — a well-meant
    /// `SINT8_PACKED_NV => S8` would otherwise be a silent widening that lets a
    /// packed-only device answer to plain `s8`.
    #[test]
    fn packed_types_are_deliberately_unmapped() {
        for raw in [
            VkComponentTypeKHR::COMPONENT_TYPE_SINT8_PACKED_NV as u32,
            VkComponentTypeKHR::COMPONENT_TYPE_UINT8_PACKED_NV as u32,
        ] {
            assert_eq!(
                component(raw),
                ComponentType::Other(raw),
                "value {raw} is unmapped on purpose; see the note on `component`. \
                 If you are mapping it, update that note and this test together."
            );
        }
    }

    /// FP8 derives from the raw device value, and derives to the **finite**
    /// spelling.
    ///
    /// The variant assertion is the cheap half. The token assertion is the one
    /// that matters: `f8e4m3fn` and `f8e4m3fnuz` differ by four characters and
    /// name different formats — different NaN handling, different exponent bias
    /// — and KISS gives the second no computation semantics at all. A device
    /// deriving the `fnuz` spelling would be claiming to compute in a format the
    /// spec says must be met with a typed decline.
    #[test]
    fn fp8_derives_from_the_raw_device_value_as_the_finite_variant() {
        for (raw, want, spelling) in [
            (
                VkComponentTypeKHR::COMPONENT_TYPE_FLOAT8_E4M3_EXT as u32,
                ComponentType::F8E4M3FN,
                "f8e4m3fn",
            ),
            (
                VkComponentTypeKHR::COMPONENT_TYPE_FLOAT8_E5M2_EXT as u32,
                ComponentType::F8E5M2,
                "f8e5m2",
            ),
        ] {
            assert_eq!(
                component(raw),
                want,
                "VkComponentTypeKHR value {raw} must derive {want:?}, not Other — \
                 otherwise FP8 is spellable but not derivable, which is the exact \
                 defect bfloat16 had"
            );

            // Spelled through a real token rather than a component-level
            // accessor: what a consumer byte-matches is the token, and the
            // vocabulary exposes no per-component spell function publicly.
            let token = VulkanTarget {
                subgroup: Subgroup::Fixed(32),
                ops: kiss_vulkan_vocab::OpClasses::NONE,
                arith: kiss_vulkan_vocab::Arith::NONE,
                coop: CoopMatrix::from_shapes(vec![kiss_vulkan_vocab::CoopShape {
                    m: 16,
                    n: 16,
                    k: 16,
                    a: want,
                    b: want,
                    c: want,
                    result: want,
                    saturating: false,
                }]),
            }
            .to_token();
            assert!(
                token.contains(spelling),
                "the derived token must spell {spelling}: {token}"
            );
            assert!(
                !token.contains("fnuz"),
                "a derived token must never carry a reserved `fnuz` spelling — \
                 KISS gives those no computation semantics: {token}"
            );
        }
    }

    /// The two NV names are registry *aliases* of the EXT enumerants, not
    /// separate values. Asserted because the alias is what makes one match arm
    /// sufficient: if a future `vk.xml` split them into distinct values, this
    /// fails and the second arm becomes necessary — where otherwise an NV-only
    /// driver would silently derive `Other`.
    #[test]
    fn the_nv_fp8_names_alias_the_ext_enumerants() {
        assert_eq!(
            VkComponentTypeKHR::COMPONENT_TYPE_FLOAT8_E4M3_EXT as u32,
            vulkane_raw_nv_e4m3(),
            "VK_COMPONENT_TYPE_FLOAT_E4M3_NV must alias FLOAT8_E4M3_EXT"
        );
        assert_eq!(
            VkComponentTypeKHR::COMPONENT_TYPE_FLOAT8_E5M2_EXT as u32,
            vulkane_raw_nv_e5m2(),
            "VK_COMPONENT_TYPE_FLOAT_E5M2_NV must alias FLOAT8_E5M2_EXT"
        );
    }

    fn vulkane_raw_nv_e4m3() -> u32 {
        crate::raw::bindings::COMPONENT_TYPE_FLOAT_E4M3_NV as u32
    }

    fn vulkane_raw_nv_e5m2() -> u32 {
        crate::raw::bindings::COMPONENT_TYPE_FLOAT_E5M2_NV as u32
    }
}
