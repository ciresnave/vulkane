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
                a: component(p.a_type() as u32),
                b: component(p.b_type() as u32),
                c: component(p.c_type() as u32),
                result: component(p.result_type() as u32),
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
/// Unknown values become [`ComponentType::Other`] rather than an error: new
/// Vulkan component types appear faster than a vocabulary revision can track
/// them, and an honest round-trippable token beats a decline.
fn component(raw: u32) -> ComponentType {
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
        n => ComponentType::Other(n),
    }
}
