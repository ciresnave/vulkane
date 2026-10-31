// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Slang parser over the whole `vulkane-kernels-source` corpus, with the same method as
//! Unpopped's `docs/slang-corpus-sizing.md` (`slang_corpus_report`): all three lifters are called
//! directly (never the combined `lift`, which reports only the last attempt's error), the declared
//! dtype is held at f32 for every file, and a file is bucketed by what the lifters said.
//!
//! The corpus is enumerated from `git ls-files`-equivalent knowledge only through the directory
//! listing filtered on the `.slang` extension, and the test prints how many non-`.slang` files
//! sit beside them, so a population mismatch is visible rather than absorbed.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use unpopped::LiftError;
use unpopped::convert::{lift_elementwise, lift_reduction, lift_scan};
use unpopped_vocab::ElementKind;
use vulkane_slang_parse::SLANG;

const F32: &[ElementKind] = &[ElementKind::F32];

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vulkane-kernels-source/kernels")
}

fn bucket(src: &str, name: &str) -> &'static str {
    let ew = lift_elementwise(&SLANG, src, name, F32);
    if ew.is_ok() {
        return "accepted: elementwise";
    }
    let rd = lift_reduction(&SLANG, src, name, F32);
    if rd.is_ok() {
        return "accepted: reduction";
    }
    let sc = lift_scan(&SLANG, src, name, F32);
    if sc.is_ok() {
        return "accepted: scan";
    }
    let errs = [ew.unwrap_err(), rd.unwrap_err(), sc.unwrap_err()];
    if errs.iter().any(|e| matches!(e, LiftError::NotAKernel)) {
        "refused: NotAKernel"
    } else if errs
        .iter()
        .any(|e| matches!(e, LiftError::Inexpressible(_)))
    {
        "refused: Inexpressible (residue)"
    } else if errs
        .iter()
        .any(|e| matches!(e, LiftError::DtypeNotAdmissible { .. }))
    {
        "refused: DtypeNotAdmissible"
    } else {
        "refused: Unrecognized"
    }
}

#[test]
fn corpus_histogram_matches_the_baseline_from_unpopped() {
    let dir = corpus_dir();
    let all: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file())
        .collect();
    let present = all.len();
    let mut slang: Vec<PathBuf> = all
        .into_iter()
        .filter(|p| p.extension().is_some_and(|x| x == "slang"))
        .collect();
    slang.sort();

    let mut hist: BTreeMap<&'static str, usize> = BTreeMap::new();
    for p in &slang {
        let src = fs::read_to_string(p).unwrap();
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        *hist.entry(bucket(&src, &name)).or_insert(0) += 1;
    }
    println!(
        "population: {} .slang files (of {present} files present)",
        slang.len()
    );
    for (k, v) in &hist {
        println!("{v:4}  {k}");
    }

    // Population: the 147 Slang kernels (positive control for the enumeration itself).
    assert_eq!(slang.len(), 147, "the corpus is the 147 .slang kernels");
    // Every file is accounted for in exactly one bucket.
    assert_eq!(hist.values().sum::<usize>(), 147);
    // Baseline measured by Unpopped (docs/slang-corpus-sizing.md, Fuel ded9bed, 2026-09-25):
    // 53 residue, 94 unrecognized, 0 accepted.
    assert_eq!(
        hist.get("refused: Inexpressible (residue)").copied(),
        Some(53)
    );
    assert_eq!(hist.get("refused: Unrecognized").copied(), Some(94));
    assert!(
        hist.keys().all(|k| k.starts_with("refused")),
        "0 accepted: {hist:?}"
    );
}
