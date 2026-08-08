//! Canonical-spelling, round-trip, and typed-decline tests for the `vulkan:`
//! capability-set vocabulary.
//!
//! The property under test throughout is the one §6.8-0002 actually needs:
//! **one target has exactly one spelling**. Round-tripping proves the spelling
//! is recoverable; the rejection tests prove no *second* spelling of the same
//! target is accepted, which matters more — byte-exact matching means two
//! accepted spellings of one target would silently fail to match each other.

use kiss_vulkan_vocab::*;

fn shape(m: u32, n: u32, k: u32, a: ComponentType, r: ComponentType) -> CoopShape {
    CoopShape {
        m,
        n,
        k,
        a,
        b: a,
        c: r,
        result: r,
        saturating: false,
    }
}

fn sample() -> VulkanTarget {
    VulkanTarget {
        subgroup: Subgroup::Fixed(32),
        ops: OpClasses::BASIC | OpClasses::ARITHMETIC | OpClasses::SHUFFLE_RELATIVE,
        arith: Arith::FLOAT16 | Arith::INT8,
        coop: CoopMatrix::from_shapes(vec![shape(
            16,
            16,
            16,
            ComponentType::F16,
            ComponentType::F32,
        )]),
    }
}

#[test]
fn spells_the_documented_shape() {
    assert_eq!(
        sample().to_token(),
        "vulkan:sg32.ops-abr.arith-f16-i8.cm-16-16-16-f16-f16-f32-f32"
    );
}

#[test]
fn round_trips() {
    let t = sample();
    assert_eq!(VulkanTarget::parse(&t.to_token()).unwrap(), t);
}

#[test]
fn round_trips_every_interesting_variant() {
    let cases = vec![
        VulkanTarget {
            subgroup: Subgroup::Dynamic,
            ops: OpClasses::NONE,
            arith: Arith::NONE,
            coop: CoopMatrix::None,
        },
        VulkanTarget {
            subgroup: Subgroup::Fixed(64),
            ops: OpClasses::ROTATE | OpClasses::ROTATE_CLUSTERED | OpClasses::PARTITIONED,
            arith: Arith::DOT8 | Arith::STORAGE8 | Arith::STORAGE16,
            coop: CoopMatrix::from_shapes(vec![
                shape(16, 16, 16, ComponentType::S8, ComponentType::S32),
                shape(8, 8, 32, ComponentType::BF16, ComponentType::F32),
                shape(16, 16, 16, ComponentType::Other(9999), ComponentType::F32),
            ]),
        },
        VulkanTarget {
            subgroup: Subgroup::Fixed(128),
            ops: OpClasses::NONE,
            arith: Arith::NONE,
            coop: CoopMatrix::Digest(0xdead_beef_0000_0001),
        },
    ];
    for t in cases {
        let tok = t.to_token();
        assert_eq!(
            VulkanTarget::parse(&tok).unwrap(),
            t,
            "round-trip failed for {tok}"
        );
    }
}

#[test]
fn saturating_shapes_are_distinct_cells() {
    let mut sat = shape(16, 16, 16, ComponentType::S8, ComponentType::S32);
    let plain = sat;
    sat.saturating = true;
    let a = CoopMatrix::from_shapes(vec![plain]);
    let b = CoopMatrix::from_shapes(vec![sat]);
    assert_ne!(a, b);
    let ta = VulkanTarget {
        subgroup: Subgroup::Fixed(32),
        ops: OpClasses::NONE,
        arith: Arith::NONE,
        coop: a,
    };
    let tb = VulkanTarget {
        coop: b,
        ..ta.clone()
    };
    assert_ne!(ta.to_token(), tb.to_token());
}

// --- the property that actually matters: no second spelling ---------------

#[test]
fn rejects_unsorted_op_letters() {
    // `ba` names the same set as `ab`. Accepting it would create two byte
    // strings for one cell, which §6.8-0002 would then fail to match.
    let e = VulkanTarget::parse("vulkan:sg32.ops-ba.arith-none.cm-none").unwrap_err();
    assert!(matches!(e, ParseError::NonCanonical { .. }), "{e:?}");
}

#[test]
fn rejects_duplicate_op_letters() {
    let e = VulkanTarget::parse("vulkan:sg32.ops-aab.arith-none.cm-none").unwrap_err();
    assert!(matches!(e, ParseError::NonCanonical { .. }), "{e:?}");
}

#[test]
fn rejects_unsorted_or_duplicated_coop_shapes() {
    let unsorted = "vulkan:sg32.ops-none.arith-none.\
                    cm-16-16-16-i8-i8-i32-i32,8-8-32-f16-f16-f32-f32";
    let e = VulkanTarget::parse(unsorted).unwrap_err();
    assert!(matches!(e, ParseError::NonCanonical { .. }), "{e:?}");

    let dup = "vulkan:sg32.ops-none.arith-none.\
               cm-8-8-32-f16-f16-f32-f32,8-8-32-f16-f16-f32-f32";
    let e = VulkanTarget::parse(dup).unwrap_err();
    assert!(matches!(e, ParseError::NonCanonical { .. }), "{e:?}");
}

#[test]
fn rejects_leading_zeros() {
    // `sg032` and `sg32` would be two spellings of one width.
    assert!(VulkanTarget::parse("vulkan:sg032.ops-none.arith-none.cm-none").is_err());
    // Same hazard inside a shape's dimensions.
    assert!(
        VulkanTarget::parse("vulkan:sg32.ops-none.arith-none.cm-016-16-16-f16-f16-f32-f32")
            .is_err()
    );
}

#[test]
fn rejects_uppercase_digest_hex() {
    let e = VulkanTarget::parse("vulkan:sg32.ops-none.arith-none.cm-fnv1a64-DEADBEEF00000001")
        .unwrap_err();
    assert!(matches!(e, ParseError::NonCanonical { .. }), "{e:?}");
}

#[test]
fn rejects_non_power_of_two_width() {
    let e = VulkanTarget::parse("vulkan:sg48.ops-none.arith-none.cm-none").unwrap_err();
    assert!(matches!(e, ParseError::NonCanonical { .. }), "{e:?}");
}

// --- §6.8-0001 / §6.8-0005 grammar and charset ----------------------------

#[test]
fn rejects_wrong_colon_count() {
    for (tok, n) in [
        ("vulkansg32.ops-none.arith-none.cm-none", 0usize),
        ("vulkan:sg32.ops-none.arith-none.cm:none", 2),
    ] {
        match VulkanTarget::parse(tok).unwrap_err() {
            ParseError::Colons { found } => assert_eq!(found, n),
            e => panic!("expected Colons for {tok}, got {e:?}"),
        }
    }
}

#[test]
fn rejects_forbidden_bytes() {
    // The structure_key field separators, plus whitespace and control bytes.
    for bad in [
        "vulkan:sg32|x",
        "vulkan:sg32;x",
        "vulkan:sg32/x",
        "vulkan:sg32 x",
    ] {
        assert!(
            matches!(
                VulkanTarget::parse(bad).unwrap_err(),
                ParseError::Charset { .. }
            ),
            "{bad} should have been rejected on charset"
        );
    }
}

#[test]
fn rejects_wrong_namespace() {
    let e = VulkanTarget::parse("cuda:sg32.ops-none.arith-none.cm-none").unwrap_err();
    assert!(matches!(e, ParseError::Namespace { .. }), "{e:?}");
}

#[test]
fn rejects_wrong_field_count() {
    assert!(VulkanTarget::parse("vulkan:sg32.ops-none.arith-none").is_err());
}

#[test]
fn never_panics_on_arbitrary_input() {
    // §7.1-0002: an unrecognized or malformed input gets a typed decline, not
    // a panic, abort, hang, or out-of-bounds read.
    let inputs = [
        "",
        ":",
        "vulkan:",
        "vulkan:...",
        "vulkan:sg.ops.arith.cm",
        "vulkan:sg-1.ops-none.arith-none.cm-none",
        "vulkan:sg99999999999999999999.ops-none.arith-none.cm-none",
        "vulkan:sgdyn.ops-zzz.arith-none.cm-none",
        "vulkan:sgdyn.ops-none.arith-none.cm-fnv1a64-",
        "vulkan:sgdyn.ops-none.arith-none.cm-1-2-3-f16",
        "vulkan:sgdyn.ops-none.arith-none.cm-,",
    ];
    for i in inputs {
        let _ = VulkanTarget::parse(i);
    }
}

// --- the digest branch ----------------------------------------------------

#[test]
fn digest_is_length_triggered_not_preferential() {
    // Grow a shape list until the encoding crosses the threshold, and assert
    // the switch happens exactly at the boundary — never earlier, never later.
    // A deriver that switched on preference rather than length would emit a
    // different token from an honest peer on the same target.
    // The threshold is measured on the canonical ENUMERATION string — the
    // comma-joined tuple list, excluding the `cm-` prefix — because that is
    // the same string the digest hashes. Measuring the prefixed field instead
    // shifts the boundary by three bytes, which a coarse step size can hide:
    // shapes here grow the enumeration ~20 bytes at a time, so an off-by-three
    // boundary would only be visible for enumerations of length 510..=512 and
    // would otherwise pass by luck. `switches_at_the_exact_byte_boundary`
    // below pins it exactly rather than relying on the sweep landing there.
    let mut shapes = Vec::new();
    let mut switched_at = None;
    for i in 1..400u32 {
        shapes.push(shape(i, 16, 16, ComponentType::F16, ComponentType::F32));
        let t = VulkanTarget {
            subgroup: Subgroup::Fixed(32),
            ops: OpClasses::NONE,
            arith: Arith::NONE,
            coop: CoopMatrix::from_shapes(shapes.clone()),
        };
        let tok = t.to_token();
        let field = tok.rsplit_once(".cm-").map(|(_, f)| f).unwrap_or("");
        let is_digest = field.starts_with("fnv1a64-");
        let enumeration_len = shapes
            .iter()
            .map(|s| format!("{}-{}-{}-f16-f16-f32-f32", s.m, s.n, s.k).len())
            .sum::<usize>()
            + (shapes.len() - 1);
        assert_eq!(
            is_digest,
            enumeration_len > COOP_DIGEST_THRESHOLD,
            "at {} shapes the enumeration is {enumeration_len} bytes; digest={is_digest}",
            shapes.len()
        );
        if is_digest && switched_at.is_none() {
            switched_at = Some(shapes.len());
        }
    }
    assert!(
        switched_at.is_some(),
        "the digest branch was never reached — it must be exercisable"
    );
}

#[test]
fn switches_at_the_exact_byte_boundary() {
    // Pins the switch between an enumeration of exactly 512 bytes and one of
    // exactly 513: `<= 512` enumerates, `> 512` digests. This is the assertion
    // two independent implementations must agree on byte for byte.
    //
    // Constructed rather than swept. A sweep that grows shapes in coarse steps
    // straddles the boundary without landing on it, so it passes even with the
    // threshold defined a few bytes off — which is exactly the discrepancy
    // that existed here while the prefixed field, rather than the bare
    // enumeration, was the measured string.
    //
    // With `n`/`k` fixed at 16, a tuple spells as `<m>-16-16-f16-f16-f32-f32`,
    // so its length is `digits(m) + 22` and the whole enumeration (tuples
    // joined by commas) is `23*N - 1 + sum(digits(m_i))`. That inverts: pick
    // `N`, solve for the digit budget, then mint distinct `m` values with
    // those digit counts.
    fn enumeration_of_len(target: usize) -> Option<Vec<CoopShape>> {
        for n in 1..=40usize {
            let need = target as isize + 1 - 23 * n as isize;
            if need < n as isize || need > 9 * n as isize {
                continue;
            }
            let mut digits = vec![1usize; n];
            let mut rem = need as usize - n;
            for d in digits.iter_mut() {
                let add = rem.min(8);
                *d += add;
                rem -= add;
                if rem == 0 {
                    break;
                }
            }
            if rem != 0 {
                continue;
            }
            // Mint distinct m values per digit width, so no two shapes collide
            // and dedup cannot silently shorten the list.
            let mut next = [0u32; 10];
            let mut shapes = Vec::new();
            for d in digits {
                let base = 10u32.pow(d as u32 - 1);
                let m = base + next[d];
                next[d] += 1;
                if m >= base.saturating_mul(10) {
                    return None;
                }
                shapes.push(shape(m, 16, 16, ComponentType::F16, ComponentType::F32));
            }
            return Some(shapes);
        }
        None
    }

    let enumeration_len = |shapes: &[CoopShape]| -> usize {
        shapes
            .iter()
            .map(|s| format!("{}-{}-{}-f16-f16-f32-f32", s.m, s.n, s.k).len())
            .sum::<usize>()
            + shapes.len().saturating_sub(1)
    };
    let is_digest = |shapes: &[CoopShape]| -> bool {
        VulkanTarget {
            subgroup: Subgroup::Fixed(32),
            ops: OpClasses::NONE,
            arith: Arith::NONE,
            coop: CoopMatrix::from_shapes(shapes.to_vec()),
        }
        .to_token()
        .contains(".cm-fnv1a64-")
    };

    let at = enumeration_of_len(COOP_DIGEST_THRESHOLD)
        .expect("could not construct an enumeration of exactly the threshold length");
    let over = enumeration_of_len(COOP_DIGEST_THRESHOLD + 1)
        .expect("could not construct an enumeration one byte over the threshold");

    assert_eq!(enumeration_len(&at), COOP_DIGEST_THRESHOLD);
    assert_eq!(enumeration_len(&over), COOP_DIGEST_THRESHOLD + 1);

    assert!(
        !is_digest(&at),
        "an enumeration of exactly {COOP_DIGEST_THRESHOLD} bytes must be spelled inline"
    );
    assert!(
        is_digest(&over),
        "an enumeration of {} bytes must be digested",
        COOP_DIGEST_THRESHOLD + 1
    );
}

#[test]
fn digest_runs_over_the_canonical_enumeration() {
    // Pinning what is hashed, not merely that hashing happens: the digest must
    // be reproducible by any implementation that can build the canonical
    // enumeration string, without reimplementing the shape-list traversal.
    let shapes: Vec<CoopShape> = (1..40)
        .map(|i| shape(i, 16, 16, ComponentType::F16, ComponentType::F32))
        .collect();
    let coop = CoopMatrix::from_shapes(shapes.clone());
    let t = VulkanTarget {
        subgroup: Subgroup::Fixed(32),
        ops: OpClasses::NONE,
        arith: Arith::NONE,
        coop,
    };
    let tok = t.to_token();
    let hex = tok.rsplit_once("fnv1a64-").expect("expected digest form").1;

    let mut sorted = shapes;
    sorted.sort();
    sorted.dedup();
    let joined = sorted
        .iter()
        .map(|s| format!("{}-{}-{}-f16-f16-f32-f32", s.m, s.n, s.k))
        .collect::<Vec<_>>()
        .join(",");
    assert_eq!(hex, format!("{:016x}", fnv1a64(joined.as_bytes())));
}

#[test]
fn fnv1a64_matches_published_vectors() {
    // The canonical FNV-1a 64 test vectors. If this ever fails, every digest
    // token this crate has emitted is wrong and incompatible with any other
    // conforming implementation.
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
}

// --- ordering invariants the macro relies on ------------------------------

#[test]
fn shape_ordering_is_total_and_stable() {
    // from_shapes must be idempotent: sorting an already-sorted list changes
    // nothing, so re-spelling a parsed token reproduces it exactly.
    let shapes = vec![
        shape(16, 16, 16, ComponentType::S8, ComponentType::S32),
        shape(8, 8, 32, ComponentType::BF16, ComponentType::F32),
        shape(16, 16, 16, ComponentType::F16, ComponentType::F32),
    ];
    let once = CoopMatrix::from_shapes(shapes.clone());
    let twice = match &once {
        CoopMatrix::Shapes(s) => CoopMatrix::from_shapes(s.clone()),
        other => other.clone(),
    };
    assert_eq!(once, twice);
}

#[test]
fn every_spelled_token_reparses() {
    // Exhaustive over op-class and arith bit patterns: any set this crate can
    // spell, it must also parse. Catches a letter that spells but has no
    // parse entry, or an ordering mismatch between the two directions.
    for ops_bits in 0u16..(1 << 11) {
        let t = VulkanTarget {
            subgroup: Subgroup::Dynamic,
            ops: OpClasses(ops_bits),
            arith: Arith::NONE,
            coop: CoopMatrix::None,
        };
        let tok = t.to_token();
        assert_eq!(VulkanTarget::parse(&tok).unwrap(), t, "ops {ops_bits:#b}");
    }
    for arith_bits in 0u8..(1 << 5) {
        let t = VulkanTarget {
            subgroup: Subgroup::Dynamic,
            ops: OpClasses::NONE,
            arith: Arith(arith_bits),
            coop: CoopMatrix::None,
        };
        let tok = t.to_token();
        assert_eq!(
            VulkanTarget::parse(&tok).unwrap(),
            t,
            "arith {arith_bits:#b}"
        );
    }
}
