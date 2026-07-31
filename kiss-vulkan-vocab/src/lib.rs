//! The `vulkan:` capability-set vocabulary for KISS-Classify §6.8
//! `target_capability` tokens.
//!
//! # What this crate is
//!
//! KISS-Classify §6.8 pins the *grammar* of a `target_capability` token —
//! `<namespace>:<capability-set>`, exactly one colon, matched **byte-exact** —
//! and delegates each namespace's capability-set vocabulary to that namespace's
//! maintainer (§6.8-0004). This crate is the reference implementation of the
//! `vulkan:` vocabulary: it spells a target canonically, parses one back, and
//! compares two.
//!
//! It is deliberately **not** the normative definition. The spelling is pinned
//! in the KISS spec text under §6.8-0004; this crate must match it byte for
//! byte, and the spec wins on any disagreement.
//!
//! # What this crate is not
//!
//! It does not talk to a GPU. **KISS-CLASSIFY-6.9-0003** forbids producing,
//! serializing, or parsing a token from loading a compute driver, kernel
//! runtime, GPU library, or backend dynamic library — an implementation must
//! manage with its standard library alone. So this crate has zero dependencies
//! and no Vulkan linkage (enforced by `tests/zero_dependency.rs`), and
//! *deriving* a token from a live `VkPhysicalDevice` lives in the `vulkane`
//! crate instead, where the Vulkan dependency already exists.
//!
//! A conformance implementation needs only this half.
//!
//! # The token names a *chosen* specialization, not a device
//!
//! A `target_capability` sits inside a `structure_key`, which identifies a
//! specialization **cell** — a kernel artifact. A wave32-pinned kernel and a
//! wave64-pinned kernel are different binaries, so they must be different
//! tokens; a token naming the device's *capability envelope* would collide
//! them on one cell. Consequently **a device admits a set of tokens rather
//! than having one**, and the consumer chooses before matching. Because
//! §6.8-0002 forbids subset and implication logic, a consumer holding a
//! 32..=64-capable device may not look up a `sg32` kernel by reasoning that
//! its envelope contains 32 — it must decide it is building a wave32 cell,
//! spell that token, and match it exactly.
//!
//! # Grammar
//!
//! ```text
//! vulkan:<subgroup>.<ops>.<arith>.<coop>
//! ```
//!
//! Four fields in fixed positions, separated by `.`; parts *within* a field
//! separated by `-`. Every set is canonically sorted and every field is
//! always present, so two independent implementations spelling the same
//! target produce identical bytes — which §6.8-0002 requires, since it gives
//! no matching tolerance whatsoever.
//!
//! | Field | Spelling | Meaning |
//! |---|---|---|
//! | subgroup | `sg32`, `sg64`, `sgdyn` | the subgroup width the kernel is built for, or `sgdyn` for width-agnostic |
//! | ops | `ops-abr`, `ops-none` | subgroup operation classes the kernel requires, canonically sorted |
//! | arith | `arith-f16-i8`, `arith-none` | arithmetic capabilities the kernel requires |
//! | coop | `cm-16-16-16-f16-f16-f32-f32`, `cm-none`, `cm-fnv1a64-<hex>` | cooperative-matrix shapes the kernel uses |
//!
//! ```
//! use kiss_vulkan_vocab::{VulkanTarget, Subgroup, OpClasses, Arith, CoopMatrix};
//!
//! let t = VulkanTarget {
//!     subgroup: Subgroup::Fixed(32),
//!     ops: OpClasses::BASIC | OpClasses::ARITHMETIC,
//!     arith: Arith::FLOAT16,
//!     coop: CoopMatrix::None,
//! };
//! assert_eq!(t.to_token(), "vulkan:sg32.ops-ab.arith-f16.cm-none");
//! assert_eq!(VulkanTarget::parse(&t.to_token()).unwrap(), t);
//! ```
//!
//! # Why `sgdyn` exists
//!
//! A kernel compiled for a fixed width and a kernel that reads
//! `gl_SubgroupSize` / `WaveGetLaneCount()` at runtime are different artifacts
//! with different performance, and both are common in practice. Without a
//! distinct spelling, every width-agnostic kernel would have to be labelled
//! with some arbitrary concrete width and would collide with the pinned
//! variant of itself.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::fmt;

/// Namespace component of every token this crate produces (§6.8-0003).
pub const NAMESPACE: &str = "vulkan";

/// Byte length at which the cooperative-matrix field switches from a full
/// canonical enumeration to a digest.
///
/// **Measured on the canonical enumeration string** — the comma-joined tuple
/// list, *excluding* the `cm-` prefix — which is deliberately the same string
/// the digest is computed over. One string with two uses means there is
/// nothing to define twice and nothing for two implementations to interpret
/// differently. The enumeration is also the only unbounded part of a token;
/// the subgroup, ops, and arith fields are bounded by their alphabets.
///
/// The switch is a **hard deterministic trigger at the exact byte count** —
/// `len <= 512` enumerates, `len > 512` digests — never an implementation
/// preference. Two honest implementations on the same target that disagreed
/// about which form to emit would produce different tokens and fail
/// §6.8-0002 byte-exact matching, which is the same determinism argument that
/// motivates the digest having a pinned hash at all. See [`CoopMatrix::spell`].
///
/// *Rationale (not normative).* 512 is `2^9`, an eighth of
/// `MAX_STRUCTURE_KEY_LEN` (4096), reserving the other seven eighths for the
/// op-family, dtype, and operand-descriptor fields so a target can never crowd
/// out the operand data that makes a `structure_key` useful. At roughly 22
/// bytes per tuple it admits about 23 shapes inline, which covers every device
/// measured so far — an AMD RDNA part reports 11, encoding to 281 bytes —
/// while keeping the digest branch reachable rather than theoretical. The
/// specific number is a policy choice; what matters for interoperation is that
/// it is pinned and identical everywhere.
pub const COOP_DIGEST_THRESHOLD: usize = 512;

/// FNV-1a 64-bit offset basis, pinned so every implementation agrees.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime, pinned so every implementation agrees.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64, over `bytes`, emitted as fixed-width lowercase hex by callers.
///
/// Pinned rather than chosen freely: §6.9-0003 requires a token be producible
/// with only a standard library, which rules out reaching for a SHA-2 crate.
/// Derivers are not adversarial here — the digest only has to avoid accidental
/// collision across the handful of shape sets real hardware reports — so a
/// non-cryptographic hash is the right tool, provided everyone uses the *same*
/// one.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Why a token could not be parsed.
///
/// Every variant is a *typed decline* in the §7.1-0002 sense: parsing an
/// unrecognized or malformed token must never panic, abort, hang, or read out
/// of bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The token did not have exactly one `:` (§6.8-0001).
    Colons {
        /// How many were found.
        found: usize,
    },
    /// The namespace was not `vulkan`.
    Namespace {
        /// What was found instead.
        found: String,
    },
    /// The capability-set did not have exactly four `.`-separated fields.
    FieldCount {
        /// How many were found.
        found: usize,
    },
    /// A field was present but not spelled canonically.
    Field {
        /// Which field, by name.
        field: &'static str,
        /// The offending text.
        found: String,
    },
    /// A byte forbidden by §6.8-0005 appeared in the token.
    Charset {
        /// The offending byte.
        byte: u8,
    },
    /// The token was spelled legally but non-canonically — e.g. an unsorted
    /// op-class set, or a duplicated cooperative-matrix shape.
    ///
    /// Rejected rather than normalized: §6.8-0002 matches byte-exact, so
    /// silently accepting a non-canonical spelling would let two spellings of
    /// one target coexist and fail to match each other.
    NonCanonical {
        /// What was wrong.
        why: &'static str,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Colons { found } => {
                write!(f, "expected exactly one ':' (§6.8-0001), found {found}")
            }
            Self::Namespace { found } => write!(f, "expected namespace `vulkan`, found `{found}`"),
            Self::FieldCount { found } => {
                write!(f, "expected 4 '.'-separated fields, found {found}")
            }
            Self::Field { field, found } => write!(f, "malformed {field} field: `{found}`"),
            Self::Charset { byte } => {
                write!(f, "byte 0x{byte:02x} is forbidden in a token (§6.8-0005)")
            }
            Self::NonCanonical { why } => write!(f, "non-canonical spelling: {why}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// The subgroup width a kernel is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Subgroup {
    /// Compiled for one specific width, pinned via
    /// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`. Always a power
    /// of two.
    Fixed(u32),
    /// Width-agnostic: reads `gl_SubgroupSize` / `WaveGetLaneCount()` at
    /// runtime, so one binary is correct at any width.
    Dynamic,
}

impl Subgroup {
    fn spell(self) -> String {
        match self {
            Self::Fixed(w) => format!("sg{w}"),
            Self::Dynamic => "sgdyn".to_string(),
        }
    }

    fn parse(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::Field {
            field: "subgroup",
            found: s.to_string(),
        };
        if s == "sgdyn" {
            return Ok(Self::Dynamic);
        }
        let digits = s.strip_prefix("sg").ok_or_else(bad)?;
        // Reject a leading zero explicitly: `sg032` and `sg32` would be two
        // spellings of one target, and §6.8-0002 would not match them.
        if digits.is_empty() || (digits.len() > 1 && digits.starts_with('0')) {
            return Err(bad());
        }
        let w: u32 = digits.parse().map_err(|_| bad())?;
        if !w.is_power_of_two() {
            return Err(ParseError::NonCanonical {
                why: "subgroup width must be a power of two",
            });
        }
        Ok(Self::Fixed(w))
    }
}

macro_rules! flag_set {
    (
        $(#[$meta:meta])* $name:ident, $repr:ty, $prefix:literal, $field:literal,
        $( $(#[$fmeta:meta])* $konst:ident = $bit:expr, $letter:literal );+ $(;)?
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $name(pub $repr);

        impl $name {
            /// The empty set.
            pub const NONE: Self = Self(0);
            $( $(#[$fmeta])* pub const $konst: Self = Self($bit); )+

            /// Every letter this set can spell, in canonical (sorted) order.
            const LETTERS: &'static [(char, $repr)] = {
                // Sorted at compile time by construction — the test
                // `letters_are_sorted_and_unique` proves it stayed that way.
                &[ $( ($letter, $bit) ),+ ]
            };

            /// Whether every bit of `other` is present.
            pub const fn contains(self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }

            /// Whether the set is empty.
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            fn spell(self) -> String {
                if self.is_empty() {
                    return format!("{}-none", $prefix);
                }
                let mut s = String::from($prefix);
                s.push('-');
                for (c, bit) in Self::LETTERS {
                    if self.0 & bit != 0 {
                        s.push(*c);
                    }
                }
                s
            }

            fn parse(s: &str) -> Result<Self, ParseError> {
                let bad = || ParseError::Field { field: $field, found: s.to_string() };
                let body = s
                    .strip_prefix($prefix)
                    .and_then(|r| r.strip_prefix('-'))
                    .ok_or_else(bad)?;
                if body == "none" {
                    return Ok(Self::NONE);
                }
                let mut acc: $repr = 0;
                let mut last: Option<char> = None;
                for ch in body.chars() {
                    let bit = Self::LETTERS
                        .iter()
                        .find(|(c, _)| *c == ch)
                        .map(|(_, b)| *b)
                        .ok_or_else(bad)?;
                    // Canonical order is the letter order, strictly ascending:
                    // duplicates and re-orderings are distinct byte strings
                    // that would name the same target, so both are rejected.
                    if let Some(p) = last {
                        if ch <= p {
                            return Err(ParseError::NonCanonical {
                                why: concat!($field, " letters must be sorted and unique"),
                            });
                        }
                    }
                    last = Some(ch);
                    acc |= bit;
                }
                Ok(Self(acc))
            }
        }

        impl std::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
        }

        impl std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
        }
    };
}

flag_set! {
    /// Subgroup operation classes a kernel requires.
    ///
    /// Letters are single ASCII characters, so the set spells unambiguously by
    /// juxtaposition — a fixed-width alphabet is uniquely decodable by
    /// construction, unlike the variable-length dtype tokens in the
    /// cooperative-matrix field, which is why *those* carry explicit
    /// separators and these do not.
    OpClasses, u16, "ops", "ops",
    /// `subgroupAdd` / `subgroupMul` / `subgroupMin` / `subgroupMax` and their
    /// scans — what a cross-lane reduction needs.
    ARITHMETIC = 1 << 0, 'a';
    /// `subgroupBarrier` / `subgroupElect` — mandatory in Vulkan 1.1.
    BASIC = 1 << 1, 'b';
    /// Clustered reductions over power-of-two lane groups.
    CLUSTERED = 1 << 2, 'c';
    /// `subgroupBallot` and friends.
    BALLOT = 1 << 3, 'l';
    /// `VK_NV_shader_subgroup_partitioned`.
    PARTITIONED = 1 << 4, 'p';
    /// Quad (2x2) shuffles and broadcasts.
    QUAD = 1 << 5, 'q';
    /// `subgroupShuffleUp` / `subgroupShuffleDown`.
    SHUFFLE_RELATIVE = 1 << 6, 'r';
    /// `subgroupShuffle` / `subgroupShuffleXor`.
    SHUFFLE = 1 << 7, 's';
    /// Clustered rotate (`VK_KHR_shader_subgroup_rotate`).
    ROTATE_CLUSTERED = 1 << 8, 't';
    /// `subgroupAll` / `subgroupAny` / `subgroupAllEqual`.
    VOTE = 1 << 9, 'v';
    /// Rotate (`VK_KHR_shader_subgroup_rotate`, Vulkan 1.4 core).
    ROTATE = 1 << 10, 'w';
}

/// Arithmetic capabilities a kernel requires.
///
/// Unlike [`OpClasses`], these spell as **named parts joined by `-`**
/// (`arith-f16-i8`) rather than juxtaposed letters. The names are
/// variable-length, so juxtaposition would depend on the set staying uniquely
/// decodable as it grows — the same fragility that makes the cooperative-matrix
/// field carry explicit separators. `OpClasses` can juxtapose safely only
/// because its alphabet is fixed-width by construction.
///
/// There are five of them and the names are short, so legibility costs almost
/// nothing here and a reader can tell what a token requires without a lookup
/// table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Arith(pub u8);

impl Arith {
    /// The empty set.
    pub const NONE: Self = Self(0);
    /// `shaderInt8` — 8-bit integer arithmetic in shaders.
    pub const INT8: Self = Self(1 << 0);
    /// Any accelerated 8-bit integer dot product
    /// (`VK_KHR_shader_integer_dot_product`), what an int8-quantized matmul
    /// actually lowers to.
    pub const DOT8: Self = Self(1 << 1);
    /// `shaderFloat16` — half-precision arithmetic in shaders.
    pub const FLOAT16: Self = Self(1 << 2);
    /// `storageBuffer8BitAccess` — 8-bit types in storage buffers.
    pub const STORAGE8: Self = Self(1 << 3);
    /// `storageBuffer16BitAccess` — 16-bit types in storage buffers.
    pub const STORAGE16: Self = Self(1 << 4);

    /// Every name this set can spell, held in **lexicographic** order so the
    /// canonical spelling is "the selected names, sorted, joined by `-`" — a
    /// rule that can be stated normatively in one line and reproduced by any
    /// implementation without consulting a bit layout.
    const NAMES: &'static [(&'static str, u8)] = &[
        ("dot8", 1 << 1),
        ("f16", 1 << 2),
        ("i8", 1 << 0),
        ("st16", 1 << 4),
        ("st8", 1 << 3),
    ];

    /// Whether every bit of `other` is present.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn spell(self) -> String {
        if self.is_empty() {
            return "arith-none".to_string();
        }
        let mut s = String::from("arith");
        for (name, bit) in Self::NAMES {
            if self.0 & bit != 0 {
                s.push('-');
                s.push_str(name);
            }
        }
        s
    }

    fn parse(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::Field {
            field: "arith",
            found: s.to_string(),
        };
        let body = s
            .strip_prefix("arith")
            .and_then(|r| r.strip_prefix('-'))
            .ok_or_else(bad)?;
        if body == "none" {
            return Ok(Self::NONE);
        }
        let mut acc = 0u8;
        let mut last: Option<&str> = None;
        for part in body.split('-') {
            let bit = Self::NAMES
                .iter()
                .find(|(n, _)| *n == part)
                .map(|(_, b)| *b)
                .ok_or_else(bad)?;
            // Strictly ascending: a re-ordered or repeated spelling names the
            // same set in different bytes, which §6.8-0002 would then fail to
            // match against the canonical form.
            if let Some(p) = last {
                if part <= p {
                    return Err(ParseError::NonCanonical {
                        why: "arith names must be sorted and unique",
                    });
                }
            }
            last = Some(part);
            acc |= bit;
        }
        Ok(Self(acc))
    }
}

impl std::ops::BitOr for Arith {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Arith {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// A cooperative-matrix component type, mirroring `VkComponentTypeKHR`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentType {
    /// 16-bit float.
    F16,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// bfloat16.
    BF16,
    /// Signed 8-bit integer.
    S8,
    /// Signed 16-bit integer.
    S16,
    /// Signed 32-bit integer.
    S32,
    /// Signed 64-bit integer.
    S64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// A type this vocabulary version does not name, carried by its raw
    /// `VkComponentTypeKHR` value.
    ///
    /// Present so a driver exposing a component type newer than this crate
    /// yields an honest, round-trippable token rather than a decline or a
    /// silent mis-spelling. New Vulkan component types appear faster than a
    /// vocabulary revision can track them.
    Other(u32),
}

impl ComponentType {
    fn spell(self) -> String {
        match self {
            Self::F16 => "f16".into(),
            Self::F32 => "f32".into(),
            Self::F64 => "f64".into(),
            Self::BF16 => "bf16".into(),
            Self::S8 => "s8".into(),
            Self::S16 => "s16".into(),
            Self::S32 => "s32".into(),
            Self::S64 => "s64".into(),
            Self::U8 => "u8".into(),
            Self::U16 => "u16".into(),
            Self::U32 => "u32".into(),
            Self::U64 => "u64".into(),
            Self::Other(n) => format!("x{n}"),
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "f16" => Self::F16,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "bf16" => Self::BF16,
            "s8" => Self::S8,
            "s16" => Self::S16,
            "s32" => Self::S32,
            "s64" => Self::S64,
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            other => {
                let n = other.strip_prefix('x')?;
                if n.is_empty() || (n.len() > 1 && n.starts_with('0')) {
                    return None;
                }
                Self::Other(n.parse().ok()?)
            }
        })
    }
}

/// One supported cooperative-matrix shape: an `M x N x K` tile with its four
/// component types, and whether accumulation saturates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoopShape {
    /// Rows of the A operand / result.
    pub m: u32,
    /// Columns of the B operand / result.
    pub n: u32,
    /// The contracted dimension.
    pub k: u32,
    /// Component type of the A operand.
    pub a: ComponentType,
    /// Component type of the B operand.
    pub b: ComponentType,
    /// Component type of the C accumulator.
    pub c: ComponentType,
    /// Component type of the result.
    pub result: ComponentType,
    /// Whether accumulation saturates.
    pub saturating: bool,
}

impl CoopShape {
    fn spell(&self) -> String {
        let mut s = format!(
            "{}-{}-{}-{}-{}-{}-{}",
            self.m,
            self.n,
            self.k,
            self.a.spell(),
            self.b.spell(),
            self.c.spell(),
            self.result.spell(),
        );
        if self.saturating {
            s.push_str("-sat");
        }
        s
    }

    fn parse(s: &str) -> Option<Self> {
        let mut parts: Vec<&str> = s.split('-').collect();
        let saturating = match parts.last() {
            Some(&"sat") => {
                parts.pop();
                true
            }
            _ => false,
        };
        if parts.len() != 7 {
            return None;
        }
        let dim = |t: &str| -> Option<u32> {
            if t.is_empty() || (t.len() > 1 && t.starts_with('0')) {
                return None;
            }
            t.parse().ok()
        };
        Some(Self {
            m: dim(parts[0])?,
            n: dim(parts[1])?,
            k: dim(parts[2])?,
            a: ComponentType::parse(parts[3])?,
            b: ComponentType::parse(parts[4])?,
            c: ComponentType::parse(parts[5])?,
            result: ComponentType::parse(parts[6])?,
            saturating,
        })
    }
}

/// The cooperative-matrix capability a kernel requires.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoopMatrix {
    /// The kernel uses no cooperative-matrix operations.
    None,
    /// The exact shapes the kernel uses, canonically sorted and deduplicated.
    Shapes(Vec<CoopShape>),
    /// A digest of a shape list too long to spell inline. See
    /// [`COOP_DIGEST_THRESHOLD`].
    Digest(u64),
}

impl CoopMatrix {
    /// Canonically sort and deduplicate a shape list.
    ///
    /// Two implementations reporting the same shapes in different driver
    /// order must still produce identical bytes, so ordering is imposed here
    /// rather than trusted from the source.
    pub fn from_shapes(mut shapes: Vec<CoopShape>) -> Self {
        if shapes.is_empty() {
            return Self::None;
        }
        shapes.sort();
        shapes.dedup();
        Self::Shapes(shapes)
    }

    /// Spell the field, switching to a digest strictly by encoded length.
    fn spell(&self) -> String {
        match self {
            Self::None => "cm-none".to_string(),
            Self::Digest(h) => format!("cm-fnv1a64-{h:016x}"),
            Self::Shapes(shapes) => {
                let joined = shapes
                    .iter()
                    .map(CoopShape::spell)
                    .collect::<Vec<_>>()
                    .join(",");
                // Threshold and hash input are the SAME string, so the rule is
                // "if the canonical enumeration exceeds N bytes, hash it" —
                // one definition, no second thing to get wrong.
                if joined.len() <= COOP_DIGEST_THRESHOLD {
                    format!("cm-{joined}")
                } else {
                    // The digest runs over the canonical enumeration string,
                    // never the raw shape list. That is what keeps the two
                    // forms faithful to one another: switching is a pure
                    // length-driven representation swap with identical input
                    // semantics, so two implementations can disagree about
                    // *whether* to hash but never about *what* is hashed.
                    format!("cm-fnv1a64-{:016x}", fnv1a64(joined.as_bytes()))
                }
            }
        }
    }

    fn parse(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::Field {
            field: "coop",
            found: s.to_string(),
        };
        let body = s.strip_prefix("cm-").ok_or_else(bad)?;
        if body == "none" {
            return Ok(Self::None);
        }
        // `fnv1a64-` cannot be confused with a shape tuple: a tuple always
        // begins with a decimal dimension.
        if let Some(hex) = body.strip_prefix("fnv1a64-") {
            if hex.len() != 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(bad());
            }
            if hex.bytes().any(|b| b.is_ascii_uppercase()) {
                return Err(ParseError::NonCanonical {
                    why: "digest hex must be lowercase",
                });
            }
            return Ok(Self::Digest(
                u64::from_str_radix(hex, 16).map_err(|_| bad())?,
            ));
        }
        let mut shapes = Vec::new();
        for t in body.split(',') {
            shapes.push(CoopShape::parse(t).ok_or_else(bad)?);
        }
        let mut sorted = shapes.clone();
        sorted.sort();
        sorted.dedup();
        if sorted != shapes {
            return Err(ParseError::NonCanonical {
                why: "cooperative-matrix shapes must be sorted and unique",
            });
        }
        Ok(Self::Shapes(shapes))
    }
}

/// A fully-specified `vulkan:` compilation target — the capability contract a
/// kernel requires of the device that runs it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VulkanTarget {
    /// The subgroup width the kernel is built for.
    pub subgroup: Subgroup,
    /// Subgroup operation classes the kernel requires.
    pub ops: OpClasses,
    /// Arithmetic capabilities the kernel requires.
    pub arith: Arith,
    /// Cooperative-matrix shapes the kernel uses.
    pub coop: CoopMatrix,
}

impl VulkanTarget {
    /// Spell this target as a canonical `target_capability` token.
    ///
    /// The output is the identity: two targets spell identically if and only
    /// if they are the same specialization cell.
    pub fn to_token(&self) -> String {
        format!(
            "{}:{}.{}.{}.{}",
            NAMESPACE,
            self.subgroup.spell(),
            self.ops.spell(),
            self.arith.spell(),
            self.coop.spell(),
        )
    }

    /// Parse a canonical token.
    ///
    /// Rejects any legal-but-non-canonical spelling rather than normalizing
    /// it: under §6.8-0002 two spellings of one target would fail to match
    /// each other, so accepting both would be worse than accepting neither.
    pub fn parse(token: &str) -> Result<Self, ParseError> {
        if let Some(b) = token
            .bytes()
            .find(|b| *b == b'|' || *b == b';' || *b == b'/' || *b <= 0x20 || *b >= 0x7f)
        {
            return Err(ParseError::Charset { byte: b });
        }
        let colons = token.bytes().filter(|b| *b == b':').count();
        if colons != 1 {
            return Err(ParseError::Colons { found: colons });
        }
        let (ns, caps) = token
            .split_once(':')
            .ok_or(ParseError::Colons { found: 0 })?;
        if ns != NAMESPACE {
            return Err(ParseError::Namespace {
                found: ns.to_string(),
            });
        }

        // `.` separates fields uniformly — always, and always into exactly
        // four. No field may contain one, which is why the digest marker is
        // spelled `fnv1a64-<hex>` rather than `fnv1a64.<hex>`: the latter
        // would force a positional "the first three dots separate" rule, and a
        // parser written to the obvious greedy reading would then accept every
        // inline token and fail only on digest-form ones — i.e. only on the
        // large devices that are hardest to test. A uniform rule needs no
        // caveat and is enforced by construction.
        let fields: Vec<&str> = caps.split('.').collect();
        if fields.len() != 4 {
            return Err(ParseError::FieldCount {
                found: fields.len(),
            });
        }
        Ok(Self {
            subgroup: Subgroup::parse(fields[0])?,
            ops: OpClasses::parse(fields[1])?,
            arith: Arith::parse(fields[2])?,
            coop: CoopMatrix::parse(fields[3])?,
        })
    }
}

impl fmt::Display for VulkanTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_token())
    }
}
